#module SCP
using LinearAlgebra
using DifferentialEquations
using ModelingToolkit, Symbolics
using SciMLSensitivity, SymbolicIndexingInterface, SciMLStructures
using ForwardDiff, ComponentArrays, DiffResults, RuntimeGeneratedFunctions
using JuMP, Clarabel
import MathOptInterface as MOI

using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary, ModelingToolkitStandardLibrary.Blocks
ModelingToolkitStandardLibrary.Blocks.get_sampled_data(a, b, c, d) = 
    ModelingToolkitStandardLibrary.Blocks.get_sampled_data(a, collect(b), convert(eltype(b), c), d)
@component function first_order_hold(; name, N, dt)
    params = @parameters vals[1:N] = zeros(N)
    systems = @named begin
        output = RealOutput()
    end
    eqs = [
        output.u ~ ModelingToolkitStandardLibrary.Blocks.get_sampled_data(t, vals, dt, false)
    ]
    return ODESystem(eqs, t, [], params; name, systems, continuous_events = [t % dt ~ 0])
end

@mtkmodel DblInt begin
    @parameters begin
        m, [tunable = false]
    end
    @variables begin
        f(t)
        x(t)
        v(t)
    end
    @equations begin
        D(v) ~ f / m
        D(x) ~ v
    end
end

function build_example_problem()
    @named dblint = DblInt()
    @named input = first_order_hold(N = 20, dt=0.05)
        
    @named model = ODESystem([dblint.f ~ input.output.u], t,
        systems = [dblint, input])
    return model # structural_simplify(model)
end

sys = build_example_problem()

RuntimeGeneratedFunctions.init(@__MODULE__)
function trajopt(
    sys, tspan, N, given_params, initial_guess,
    ic, running_cost, terminal_cost, 
    g, h, Ph
)
    t = sys.iv
    (ti, tf) = tspan
    dtime = tf - ti
    augmenting_vars = ModelingToolkit.@variables begin
        l(t)=0
        y(t)=0
    end
    eqs = [
        D(l) ~ running_cost,
        D(y) ~ sum(max.(0.0, g) .^ 2) + sum(h .^ 2)
    ]
    augmented_system = ODESystem(eqs, t, systems=[sys], name=:augmented_system)
    tsys = structural_simplify(augmented_system)
    terminal_cost_fun = @RuntimeGeneratedFunction(generate_custom_function(tsys, terminal_cost))
    terminal_cstr_fun = @RuntimeGeneratedFunction(generate_custom_function(tsys, Ph))
    params = ModelingToolkit.MTKParameters(tsys, given_params)
    tunable, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), params)
    upd_start = setu(tsys, unknowns(tsys))

    get_cost = getu(tsys, tsys.l)
    upd_cost = setu(tsys, tsys.l)

    iguess = ModelingToolkit.varmap_to_vars(initial_guess, unknowns(tsys); defaults=Dict(unknowns(tsys) .=> (zeros(N), )), promotetoconcrete=false)
    iguess = collect(reduce(hcat, iguess)')
    base_prob = ODEProblem(tsys, unknowns(tsys) .=> iguess[:, 1], (0.0, 1.0), given_params; dtmax = 0.01)
    nunk = length(unknowns(tsys))

    function linearize(inp)
        neltype = eltype(inp)
        params_ = SciMLStructures.replace(SciMLStructures.Tunable(), params, inp.params)
        function segment(prob, i, repeat)
            nu0 = neltype.(prob.u0)
            upd_start(nu0, inp.u0[:, i])
            return remake(prob, u0=nu0, p=params_, tspan=((i-1)/(N-1), (i)/(N-1)) .* dtime .+ ti)
        end
        ensemble = EnsembleProblem(base_prob, prob_func=segment, safetycopy=false)
        sim = solve(ensemble, Tsit5(), trajectories=N-1)
        final_state = collect(sim[end].u[end]) 
        upd_cost(final_state, get_cost(final_state) + terminal_cost_fun(sim[end].u, params, sim[end].t))
        terminal_cstr_value = terminal_cstr_fun(final_state, params, sim[end].t)
        #@show terminal_cstr_value
        return [reduce(vcat, map(s->s.u[end], sim[1:end-1])); final_state; terminal_cstr_value]
    end

    function linearize(states, pars)
        linpoint = ComponentArray(u0=states, params=pars)
        res = DiffResults.JacobianResult(zeros(nunk * (N-1) + 1), linpoint);
        ForwardDiff.jacobian!(res, linearize, linpoint)
        return res
    end

    tic = ModelingToolkit.varmap_to_vars(ic, unknowns(tsys); defaults=Dict([l => 0.0, y => 0.0]))
    xref = iguess
    uref = tunable
    uhist = []
    xhist = []
    whist = []
    costs = []
    rcosts = []
    nparams = length(tunable)
    res = linearize(xref[:, 1:N-1], uref)

    r = 1.0
    β = 2.0
    α = 2.0
    ρ₀ = 0.0
    ρ₁ = 0.25
    ρ₂ = 0.7

    last_cost = 0#abs(res.value[end]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end])
    for i=1:20

        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false)
        @variable(model, δx[1:nunk,1:N])
        @variable(model, w[1:nunk,1:N-1])
        @variable(model, δu[1:nparams])
        @variable(model, tc_lin)
        @constraint(model, [reshape(δx[:, 2:N], :) .+ reshape(w, :); tc_lin] .== res.derivs[1] * [reshape(δx[:, 1:(N-1)], :); δu] .+ res.value .- [reshape(xref[:, 2:N], :); 0])
        @constraint(model, reshape(δx[:, 1], :) .== tic .- reshape(xref[:, 1], :))
        @constraint(model, reshape(δx[3:4, end], :) .== [0.0, 1.0] .- reshape(xref[3:4, end], :))
        # TODO
        @constraint(model, δu + uref .<= 1.0)
        @constraint(model, -1.0 .<= δu + uref)
        
        @variable(model, μ)
        @constraint(model, [μ; reshape(w, :)] ∈ MOI.NormOneCone(length(reshape(w, :)) + 1))
        
        @variable(model, ηₚ)
        @constraint(model, [ηₚ; 1.0; reshape(δx, :); reshape(δu, :)] ∈ MOI.RotatedSecondOrderCone(length(reshape(δx, :)) + length(reshape(δu, :)) + 2))

        
        @variable(model, ν >= 0)
        @constraint(model, [ν; tc_lin] ∈ MOI.NormOneCone(2))

        @variable(model, L)
        @constraint(model, L == get_cost(δx[:, N]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end]))
        @objective(model, Min, 1000*μ + r*ηₚ + 10*ν + L)
        optimize!(model)

        est_cost = 1000*value(μ) + value(ν) + value(L) # the linearized cost estimate from the last iterate
        xref_candidate = xref .+ value.(δx)
        uref_candidate = uref .+ value.(δu)
        res_candidate = linearize(xref[:, 1:N-1], uref)
        actual_cost = 1000*norm(res_candidate.value[1:end-1] .- reshape(xref[:, 2:N], :) .- reshape(value.(δx)[:, 2:N], :), 1) + abs(res_candidate.value[end]) + get_cost(reshape(res_candidate.value[1:end-1], nunk, N-1)[:, end])
        push!(costs, est_cost)
        push!(rcosts, actual_cost)
        dk = last_cost - actual_cost
        dl = last_cost - est_cost

        ρᵏ = dk/dl
        @show ρᵏ actual_cost est_cost last_cost
        if ρᵏ < ρ₀ # it's gotten worse by too much, increase the penalty by α
            r *= α
            println("REJECT $r try $(r*α)")
            sleep(0.5)
            continue # reject the step
        end
        res = res_candidate # accept the step
        xref = xref_candidate
        uref = uref_candidate
        if ρᵏ < ρ₁# it's gotten worse by too much (but acceptable), increase the penalty by α
            println("OK, CONTRACT $r to $(r*α)")
            r *= α
        elseif ρᵏ < ρ₂
            println("OK $r")
            # it's FINE go again 
        else
            println("OK, EXPAND $r TO $(r/β)")
            # it hasn't gotten good enough decrease the penalty by β
            r /= β
        end
        push!(uhist, uref)
        push!(xhist, xref)
        push!(whist, value.(w))
        last_cost = actual_cost
        sleep(0.5)
    end
    return (uhist, xhist, whist, costs, rcosts, linearize)
end

u,x,wh,ch,rch,lnz = trajopt(sys, (0.0, 1.0), 20, Dict(sys.dblint.m => 0.25), Dict(sys.dblint.x => collect(LinRange(0.0, 0.4, 20)), sys.dblint.v => zeros(20)), 
    [sys.dblint.x => 0.0, sys.dblint.v => 0.0], (sys.dblint.f) .^ 2, 0.0, 
    sys.dblint.x - 3.0, 0.0, 500*(sys.dblint.v)^2 + 500*(10*(sys.dblint.x - 1.0))^2)
    
using Makie, GLMakie
f = Figure()
ax1 = Makie.Axis(f[1,1])
for xref in x[end-1:end]
    lines!(ax1, 1:length(xref[end, :]), xref[end-1, :])
end
ax2 = Makie.Axis(f[2,1])
for xref in x[end-1:end]
    lines!(ax2, 1:length(xref[end, :]), xref[end, :])
end
ax3 = Makie.Axis(f[3,1])
for uref in u[end-1:end]
    lines!(ax3, 1:length(uref), uref)
end
f


csys = structural_simplify(sys)
prob = ODEProblem(csys, [csys.dblint.m => 0.25, csys.dblint.v => 0.0, csys.dblint.x => 0.0, csys.input.vals => u[end]], (0.0, 1.0))
sol = solve(prob, Tsit5(); dtmax=0.01)
lines(sol.t, sol[csys.dblint.x])
lines(Float64.(1:length(xref[end, :])), xref[end, :])

function linearize(ref)
    neltype = eltype(ref)
    prob = ODEProblem(sys, [sys.dblint.v => 0.0, sys.dblint.x=>0.0, sys.dblint.m => 1.0], (0.0, 1.0))
    params = ModelingToolkit.MTKParameters(sys, [sys.dblint.m => 1.0])   
    tunable, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), params)
    params_ = SciMLStructures.replace(SciMLStructures.Tunable(), params, neltype.(tunable))
    upd_inp = setp(sys, sys.input.vals)
    upd_start = setu(sys, [sys.dblint.v, sys.dblint.x])

    function segment(prob, i, repeat)
        neltype = eltype(ref)
        nu0 = neltype.(prob.u0)
        upd_start(nu0, ref.u0[:, i])
        params_ = copy(params_)
        upd_inp(params_, ref.p)
        return remake(prob, u0=nu0, p=params_, tspan=((i-1)/10, i/10))
    end
    ensemble = EnsembleProblem(prob, prob_func=segment)
    sim = solve(ensemble, Tsit5(), trajectories=20)
    return reduce(vcat, map(s->s.u[end], sim))
end
@time sol = linearize(ComponentArray(u0=zeros(2,20), p=ones(20)))

res = DiffResults.JacobianResult(zeros(40), ComponentArray(u0=zeros(2,20), p=ones(20)));
transition_system = @time ForwardDiff.jacobian(linearize, ComponentArray(u0=zeros(2,20), p=ones(20)))
ForwardDiff.jacobian!(res, linearize, ComponentArray(u0=zeros(2,20), p=zeros(20)))
using JuMP, ECOS, Clarabel

xref = zeros(2, 21)
uref = zeros(20)
ic = zeros(2)
tc = [0.0, 1.0 - 1e-2]
#model = Model(ECOS.Optimizer)
model = Model(Clarabel.Optimizer)
@variable(model, δx[1:2,1:21])
@variable(model, δu[1:20])
@constraint(model, reshape(δx[:, 2:21], :) .== res.derivs[1] * [reshape(δx[:, 1:20], :); δu] .+ res.value .- reshape(xref[:, 2:21], :))
@constraint(model, reshape(δx[:, 1], :) .== ic .- reshape(xref[:, 1], :))
@constraint(model, reshape(δx[:, 21], :) .== tc .- reshape(xref[:, 21], :))
@constraint(model, δu + uref .<= 1.0)
@constraint(model, -1.0 .<= δu + uref)

@variable(model, μ)
@constraint(model, [μ; reshape(δx[:, 2:21], :) .+ reshape(xref[:, 2:21], :) .- res.value] ∈ MOI.NormOneCone(length(reshape(δx[:, 2:21], :)) + 1))

@variable(model, ηₚ)
@constraint(model, [ηₚ; 1.0; reshape(δx, :); reshape(δu, :)] ∈ MOI.RotatedSecondOrderCone(length(reshape(δx, :)) + length(reshape(δu, :)) + 2))

@variable(model, ηᵤ)
@constraint(model, [ηᵤ; 1.0; δu .+ uref] ∈ MOI.RotatedSecondOrderCone(length(δu) + 2))

@objective(model, Min, 10*μ + 100*ηₚ + ηᵤ)
optimize!(model)
value.(μ)

ForwardDiff.jacobian!(res, linearize, ComponentArray(u0=xref[:, 1:20] + value.(δx[:, 1:20]), p=uref + value.(δu)))
xref = xref + value.(δx)
uref = uref + value.(δu)
