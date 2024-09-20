#module SCP
using LinearAlgebra
using OrdinaryDiffEq
using ModelingToolkit, Symbolics, Setfield
using SciMLSensitivity, SymbolicIndexingInterface, SciMLStructures, IntervalSets
using ForwardDiff, ComponentArrays, DiffResults, RuntimeGeneratedFunctions
using JuMP, Clarabel
import MathOptInterface as MOI

using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary, ModelingToolkitStandardLibrary.Blocks


struct RelevantTime end
Symbolics.option_to_metadata_type(::Val{:relevant_time}) = RelevantTime
function reltime(x, default = (-Inf,Inf))
    p = Symbolics.getparent(x, nothing)
    p === nothing || (x = p)
    Symbolics.getmetadata(x, RelevantTime, default)
end

Symbolics.@register_symbolic get_sampled_data_internal(t::Float64, buffer::Vector{Float64}, dt, circular_buffer)
function get_sampled_data_internal(a, b, c, d)
    ModelingToolkitStandardLibrary.Blocks.get_sampled_data(a, collect(b), convert(eltype(b), c), d)
end
@component function first_order_hold(; name, N, dt)
    syms = [Symbol("val$i") for i=1:N]
    params = [first(@parameters $sym = 0.0, [tunable = true, relevant_time = ((i-1)*dt,i*dt)]) for (i, sym) in enumerate(syms)]
    @parameters vals[1:N]=zeros(N) 
    systems = @named begin
        output = RealOutput()
    end
    eqs = [
        output.u ~ ifelse(t < dt*N, get_sampled_data_internal(t, vals, dt, false), params[end])
    ]
    pdeps = [vals ~ params]
    return ODESystem(eqs, t, [], [[vals]; params]; name, systems, continuous_events = [t % dt ~ 0], parameter_dependencies = pdeps)
end

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
        final_state = collect(last(sim)[end]) 
        upd_cost(final_state, get_cost(final_state) + terminal_cost_fun(inp.u0[:,N], params, last(sim).t))
        terminal_cstr_value = terminal_cstr_fun(inp.u0[:,N], params, last(sim).t)
        #@show terminal_cstr_value
        return [reduce(vcat, map(s->s.u[end], sim[1:end-1])); final_state; terminal_cstr_value]
    end

    function linearize(states, pars)
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
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
    delta_lin_hist = []
    nparams = length(tunable)
    linearize(ComponentArray(u0=collect(xref[:, 1:N]), params=collect(uref)))
    res = linearize(xref[:, 1:N], uref)
    rd = collect(res.derivs[1]) # res gets clobbered for some reason by linearize?
    rv = collect(res.value)

    r = 1.0
    β = 2.0
    α = 2.0
    ρ₀ = 0.0
    ρ₁ = 0.25
    ρ₂ = 0.7

    last_cost = Inf #abs(res.value[end]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end])
    for i=1:100

        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false)
        @variable(model, δx[1:nunk,1:N])
        @variable(model, w[1:nunk,1:N-1])
        @variable(model, δu[1:nparams])
        @variable(model, tc_lin)
        @constraint(model, [reshape(δx[:, 2:N], :) .+ reshape(w, :); tc_lin] .== rd * [reshape(δx[:, 1:N], :); δu] .+ rv .- [reshape(xref[:, 2:N], :); 0])
        @constraint(model, reshape(δx[:, 1], :) .== tic .- reshape(xref[:, 1], :))
        @constraint(model, reshape(δx[3:4, end], :) .== [0.0, 1.0] .- reshape(xref[3:4, end], :))
        # TODO
        @constraint(model, δu + uref .<= 1.0)
        @constraint(model, -1.0 .<= δu + uref)
        
        @variable(model, μ)
        @constraint(model, [μ; reshape(w, :)] ∈ MOI.NormOneCone(length(reshape(w, :)) + 1))
        
        @variable(model, ηₚ)
        @constraint(model, [ηₚ; 1.0; reshape(δx, :); reshape(δu, :)] ∈ MOI.RotatedSecondOrderCone(length(reshape(δx, :)) + length(reshape(δu, :)) + 2))

        #=
        @variable(model, ηₙ) # terminal constraint trust region
        @constraint(model, [ηₙ; 1.0; tc_lin - res.value[end]] ∈ MOI.RotatedSecondOrderCone(3))
=#
        @variable(model, ν >= 0)
        @constraint(model, [ν; tc_lin] ∈ MOI.NormOneCone(2))

        @variable(model, L)
        @constraint(model, L == get_cost(δx[:, N]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end]))
        @objective(model, Min, 1000*μ + r*ηₚ +500*ν + L)
        optimize!(model)

        est_cost = 1000*value(μ) + 500*value(ν) + value(L) # the linearized cost estimate from the last iterate
        xref_candidate = xref .+ value.(δx)
        uref_candidate = uref .+ value.(δu)
        res_candidate = linearize(xref_candidate, uref_candidate)
        @show value(μ) value(ν) value(L) value(ηₚ) 0.5*sum(value.([reshape(δx, :); reshape(δu, :)])).^2
        push!(delta_lin_hist, res_candidate.derivs[1])
        #@show predicted
        actual = res_candidate.value[1:end-1]
        #@show actual
        lin_err = actual .- reshape(xref_candidate[:, 2:N], :)
        #@show lin_err
        actual_cost = 1000*norm(lin_err, 1) + 500*abs(res_candidate.value[end]) + get_cost(reshape(res_candidate.value[1:end-1], nunk, N-1)[:, end])
        push!(costs, est_cost)
        push!(rcosts, actual_cost)
        dk = last_cost - actual_cost
        dl = last_cost - est_cost

        ρᵏ = dk/dl
        @show norm(lin_err, 1) abs(res_candidate.value[end]) get_cost(reshape(res_candidate.value[1:end-1], nunk, N-1)[:, end])
        @show ρᵏ actual_cost est_cost last_cost
        if ρᵏ < ρ₀ # it's gotten worse by too much, increase the penalty by α
            println("REJECT $r try $(r*α)")
            r *= α
            #sleep(0.5)
            continue # reject the step
        end
        
        if maximum(abs.(xref .- xref_candidate)) < 1e-4 && maximum(abs.(uref .- uref_candidate)) < 1e-4
            break # done
        end
        res = res_candidate # accept the step
        rd = collect(res.derivs[1]) # res gets clobbered for some reason by linearize?
        rv = collect(res.value)
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
            # it hasn't gotten good enough decrease the penalty by a factor of β
            r /= β
        end
        push!(uhist, uref)
        push!(xhist, xref)
        push!(whist, value.(w))
        last_cost = actual_cost
        #sleep(0.5)
    end
    return (uhist, xhist, whist, costs, rcosts, delta_lin_hist, linearize)
end
