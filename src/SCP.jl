#module SCP
using LinearAlgebra
using OrdinaryDiffEq
using ModelingToolkit, Symbolics, Setfield
using SciMLSensitivity, SymbolicIndexingInterface, SciMLStructures, IntervalSets
using ForwardDiff, ComponentArrays, DiffResults, RuntimeGeneratedFunctions
using JuMP, Clarabel, Ipopt
import MathOptInterface as MOI

using SparseConnectivityTracer, SparseDiffTools

using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkitStandardLibrary, ModelingToolkitStandardLibrary.Blocks

struct TimeDilation end 
Symbolics.option_to_metadata_type(::Val{:dilation}) = TimeDilation
function isdilation(x, default = false)
    p = Symbolics.getparent(x, nothing)
    p === nothing || (x = p)
    Symbolics.getmetadata(x, TimeDilation, default)
end

struct RelevantTime end
Symbolics.option_to_metadata_type(::Val{:relevant_time}) = RelevantTime
function reltime(x, default = (-Inf,Inf))
    p = Symbolics.getparent(x, nothing)
    p === nothing || (x = p)
    Symbolics.getmetadata(x, RelevantTime, default)
end



function ModelingToolkitStandardLibrary.Blocks.linear_interpolation(x1::SparseConnectivityTracer.GradientTracer, x2::Real, t1::Real, t2::Real, t)
    if t1 != t2
        slope = (x2 - x1) / (t2 - t1)
        intercept = x1 - slope * t1

        return slope * t + intercept
    else
        return x2
    end
end
function ModelingToolkitStandardLibrary.Blocks.get_sampled_data(t,
        buffer::AbstractArray{T},
        dt,
        circular_buffer = true) where {T <: Real,}
    if t < 0
        t = zero(t)
    end

    if isempty(buffer)
        if T <: AbstractFloat
            return T(NaN)
        else
            return zero(T)
        end
    end

    i1 = floor(Int, t / dt) + 1 #expensive
    i2 = i1 + 1

    t1 = (i1 - 1) * dt
    x1 = @inbounds buffer[i1]

    if t == t1
        return x1
    else
        n = length(buffer)

        if circular_buffer
            i1 = (i1 - 1) % n + 1
            i2 = (i2 - 1) % n + 1
        else
            if i2 > n
                i2 = n
                i1 = i2 - 1
            end
        end

        t2 = (i2 - 1) * dt
        x2 = @inbounds buffer[i2]
        return ModelingToolkitStandardLibrary.Blocks.linear_interpolation(x1, x2, t1, t2, t)
    end
end
Symbolics.@register_symbolic get_sampled_data_internal(t::Float64, buffer::Vector{Float64}, dt, circular_buffer)
function get_sampled_data_internal(a, b, c, d)
    ModelingToolkitStandardLibrary.Blocks.get_sampled_data(a, (b), c #= convert(eltype(b), c) =#, d)
end
@component function first_order_hold(; name, N, dt)
    syms = [Symbol("val$i") for i=1:N]
    #params = [first(@parameters $sym = 0.0, [tunable = true, relevant_time = ((i-1)*dt,i*dt)]) for (i, sym) in enumerate(syms)]
    @parameters vals[1:N]=zeros(N) 
    systems = @named begin
        output = RealOutput()
    end
    eqs = [
        output.u ~ ifelse(t < dt*N, get_sampled_data_internal(t, vals, dt, false), vals[end])
    ]
    #pdeps = [vals ~ params]
    return ODESystem(eqs, t, [], [vals]; name, systems #=continuous_events = [t % dt ~ 0],=#)
end

RuntimeGeneratedFunctions.init(@__MODULE__)


denamespace(sys, x::AbstractVector) = map(v -> denamespace(sys, v), x)
function denamespace(sys, x)
    x = Symbolics.unwrap(x)
    if Symbolics.iscall(x)
        if operation(x) isa ModelingToolkit.Operator
            return Symbolics.maketerm(typeof(x), operation(x),
                Any[denamespace(sys, arg) for arg in arguments(x)],
                Symbolics.metadata(x))
        end
        if operation(x) === getindex
            args = arguments(x)
            return Symbolics.maketerm(
                typeof(x), operation(x), vcat(denamespace(sys, args[1]), args[2:end]),
                Symbolics.metadata(x))
        end
        if operation(x) isa Function
            return Symbolics.maketerm(typeof(x), operation(x),
                Any[denamespace(sys, arg) for arg in arguments(x)],
                Symbolics.metadata(x))
        end
        dns_op = denamespace(sys, operation(x))
        if isnothing(dns_op) return nothing end
        return Symbolics.maketerm(typeof(x), dns_op, arguments(x), Symbolics.metadata(x))
    elseif x isa Symbolics.Symbolic
        if !startswith(string(x), string(ModelingToolkit.get_name(sys)))
            return ParentScope(x)
        end
        new_name = strip(chopprefix(string(x), string(ModelingToolkit.get_name(sys))), [ModelingToolkit.NAMESPACE_SEPARATOR])
        Symbolics.rename(x, Symbol(new_name))
    else 
        return x
    end
end
function substitute_namespaced(sys::ModelingToolkit.AbstractSystem, rules::Union{Vector{<:Pair}, Dict})
    if ModelingToolkit.has_continuous_domain(sys) && ModelingToolkit.get_continuous_events(sys) !== nothing &&
       !isempty(ModelingToolkit.get_continuous_events(sys)) ||
       ModelingToolkit.has_discrete_events(sys) && ModelingToolkit.get_discrete_events(sys) !== nothing &&
       !isempty(ModelingToolkit.get_discrete_events(sys))
        @warn "`substitute` only supports performing substitutions in equations. This system has events, which will not be updated."
    end
    if ModelingToolkit.keytype(eltype(rules)) <: Symbol
        dict = ModelingToolkit.todict(rules)
        systems = ModelingToolkit.get_systems(sys)
        # post-walk to avoid infinite recursion
        @set! sys.systems = map(sys->substitute_namespaced(sys, dict), systems)
        ModelingToolkit.something(get(rules, nameof(sys), nothing), sys)
    elseif sys isa ODESystem
        rules = ModelingToolkit.todict(filter(r->!isnothing(r[1]), map(r -> denamespace(sys, Symbolics.unwrap(r[1])) => denamespace(sys, Symbolics.unwrap(r[2])), collect(rules))))
        @show rules ModelingToolkit.get_name(sys)
        eqs = expand_derivatives.(ModelingToolkit.fast_substitute(ModelingToolkit.get_eqs(sys), rules))
        pdeps = ModelingToolkit.fast_substitute(ModelingToolkit.get_parameter_dependencies(sys), rules)
        defs = Dict(ModelingToolkit.fast_substitute(k, rules) => ModelingToolkit.fast_substitute(v, rules)
        for (k, v) in ModelingToolkit.get_defaults(sys))
        guess = Dict(ModelingToolkit.fast_substitute(k, rules) => ModelingToolkit.fast_substitute(v, rules)
        for (k, v) in ModelingToolkit.get_guesses(sys))
        subsys = map(s -> substitute_namespaced(s, rules), ModelingToolkit.get_systems(sys))
        @show expand_derivatives.(eqs)
        ODESystem(eqs, ModelingToolkit.get_iv(sys); name = nameof(sys), defaults = defs,
            guesses = guess, parameter_dependencies = pdeps, systems = subsys)
    else
        error("substituting symbols is not supported for $(typeof(sys))")
    end
end




function trajopt(
    sys, tspan, N, given_params, initial_guess,
    ic, running_cost, terminal_cost, 
    g, h, Ph,
    convex_mod=nothing
)
@show g h Ph
@show terminal_cost
    t = sys.iv
    (ti, tf) = tspan
    dtime = tf - ti
    augmenting_vars = ModelingToolkit.@variables begin
        l(t)=0
        y(t)=0
    end
    eqs = [
        D(l) ~ running_cost,
        D(y) ~ sum((max.(0.0, g)) .^ 2) + sum(h .^ 2)
    ]
    @show running_cost
    augmented_system = ODESystem(eqs, t, systems=[sys], name=:augmented_system)
    exsys = structural_simplify(augmented_system)
    tsys = exsys
    #=
    augsys = complete(augmented_system)
    @parameters α[1:length(unknowns(exsys))]
    augsys = expand_derivatives.(substitute.(equations(exsys), (Dict(Symbolics.scalarize(unknowns(exsys) .=> α .* unknowns(exsys))), )))
    for eq in augsys
    @show eq
    end
    throw("hi")
    =#

    terminal_cost_fun = @RuntimeGeneratedFunction(generate_custom_function(tsys, terminal_cost))
    terminal_cstr_fun = @RuntimeGeneratedFunction(generate_custom_function(tsys, Ph))

    get_cost = getu(tsys, tsys.l)
    upd_cost = setu(tsys, tsys.l)

    iguess = ModelingToolkit.varmap_to_vars(initial_guess, unknowns(tsys); defaults=Dict(unknowns(tsys) .=> (zeros(N), )), promotetoconcrete=false)
    @show iguess unknowns(tsys) initial_guess
    iguess = collect(reduce(hcat, iguess)')
    base_prob = ODEProblem(tsys, unknowns(tsys) .=> iguess[:, 1], (0.0, 1.0), given_params; dtmax = 0.01)
    nunk = length(unknowns(tsys))
    
    params = parameter_values(base_prob)
    tunable, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), params)
    upd_start = setu(tsys, unknowns(tsys))

    tparams = tunable_parameters(tsys)
    dil = isdilation.(tparams)

    function lnz(;opts...)
        return function (inp)
            neltype = eltype(inp)
            params_ = SciMLStructures.replace(SciMLStructures.Tunable(), params, inp.params)
            function segment(prob, i, repeat)
                nu0 = neltype.(prob.u0)
                upd_start(nu0, inp.u0[:, i])
                return remake(prob, u0=nu0, p=params_, tspan=((i-1)/(N-1), (i)/(N-1)) .* dtime .+ ti)
            end
            ensemble = EnsembleProblem(base_prob, prob_func=segment, safetycopy=false)
            sim = solve(ensemble, Tsit5(), trajectories=N-1;save_everystep=false, opts...)
            final_state = collect(last(sim)[end]) 
            upd_cost(final_state, get_cost(final_state) + terminal_cost_fun(inp.u0[:,N], params_, last(sim).t))
            terminal_cstr_value = terminal_cstr_fun(inp.u0[:,N], params, last(sim).t)
            #@show terminal_cstr_value
            return [reduce(vcat, map(s->s.u[end], sim[1:end-1])); final_state; terminal_cstr_value]
        end
    end
    linearize(inp) = lnz()(inp)


    function sparsity_linearize(states, pars)
        detector = TracerSparsityDetector();
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        res = zeros(nunk * (N-1) + 1);
        Float64.(jacobian_sparsity(lnz(adaptive=false, dt=0.001, unstable_check=(dt,u,p,t) -> false),linpoint,detector))
    end

    #sparsity_pattern = sparsity_linearize(collect(iguess[:, 1:N]), collect(tunable))
    #colorvec = matrix_colors(sparsity_pattern)

    function linearize(states, pars)
#=
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        value = collect(linearize(linpoint))
=#
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        res = DiffResults.JacobianResult(zeros(nunk * (N-1) + 1), linpoint);
        ForwardDiff.jacobian!(res, linearize, linpoint)
        #=
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        value2 = collect(linearize(linpoint))
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        jac = forwarddiff_color_jacobian(lnz(), linpoint, colorvec = colorvec, sparsity=sparsity_pattern)
        global result_sparse = jac
        global result_dense = res.derivs[1]

        global value_dense = res.value
        global value_sparse = value
        global value_sparse2 = value2
        =#
        return res
        
        #return (value=value, derivs=(jac, ))
    end

    return Dict(
        [:ic => ic, 
        :tsys => tsys, 
        :avars => [l, y],
        :linearize => linearize, 
        :iguess => iguess,
        :tunable => tunable,
        :get_cost => get_cost,
        :N => N,
        :dil => dil, 
        :convex_mod => convex_mod
        #:jac_sparsity => sparsity_pattern,
        #:colorvec => colorvec
        ])
end
function do_trajopt(prb; maxsteps=300)
    ic = prb[:ic]
    tsys = prb[:tsys]
    l, y = prb[:avars]
    linearize = prb[:linearize]
    iguess = prb[:iguess]
    tunable = prb[:tunable]
    get_cost = prb[:get_cost]
    convex_mod = prb[:convex_mod]
    N = prb[:N]
    dil = prb[:dil]
    nunk = length(unknowns(tsys))


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
    #return res
    rd = collect(res.derivs[1]) # res gets clobbered for some reason by linearize?
    rv = collect(res.value)

    r = 1.0
    β = 2.0
    α = 2.0
    ρ₀ = 0.0
    ρ₁ = 0.25
    ρ₂ = 0.7

    last_cost = Inf #abs(res.value[end]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end])
    @show tic
    for i=1:maxsteps
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false)
        @variable(model, δx[1:nunk,1:N])
        @variable(model, w[1:nunk,1:N-1])
        @variable(model, δu[1:nparams])
        @variable(model, tc_lin)
        @constraint(model, [reshape(δx[:, 2:N], :) .+ reshape(w, :); tc_lin] .== rd * [reshape(δx[:, 1:N], :); δu] .+ rv .- [reshape(xref[:, 2:N], :); 0])
        @constraint(model, reshape(δx[:, 1], :) .== tic .- reshape(xref[:, 1], :))
        #@constraint(model, reshape(δx[3:6, end], :) .== [0.0, 0.0, 1.0, 1.0] .- reshape(xref[3:6, end], :))
        # TODO
        for i=1:nparams
            if dil !== nothing && i < length(dil) && dil[i]
                continue 
            end
            @constraint(model, δu[i] + uref[i] <= 1.0)
            @constraint(model, -1.0 <= δu[i] + uref[i])
        end

        if !isnothing(convex_mod)
            convex_mod(model, δx, xref, δu, uref)
        end
        @constraint(model, δx[4:5,N] .+ xref[4:5,N] .== [0.0,0.0]) # omega
        @constraint(model, δx[6:7,N] .+ xref[6:7,N] .== [0.0,0.0]) # R
        #@constraint(model, δx[11:13,N] .+ xref[11:13,N] .== [0.0,0.0,0.0]) # v
        @constraint(model, δx[8:9,N] .+ xref[8:9,N] .== [0.0,0.0]) # v[1:2]
        @constraint(model, δx[11:13,N] .+ xref[11:13,N] .== [0.0,0,0.0]) # pos
        
        @variable(model, μ)
        @constraint(model, [μ; reshape(w, :)] ∈ MOI.NormOneCone(length(reshape(w, :)) + 1))
        
        @variable(model, ηₚ)
        @constraint(model, [ηₚ; 1.0; reshape(δx, :); reshape(δu, :)] ∈ MOI.RotatedSecondOrderCone(length(reshape(δx, :)) + length(reshape(δu, :)) + 2))
        
        @variable(model, ηₗ)
        @constraint(model, ηₗ>=0.0)
        
        for i=1:nparams
            if dil !== nothing && i < length(dil) && dil[i]
                @show i
                @constraint(model, δu[i] <= ηₗ)
                @constraint(model, -ηₗ <= δu[i])
                #@constraint(model, 0 <= δu[i] + uref[i])
            end
        end

        #=
        @variable(model, ηₙ) # terminal constraint trust region
        @constraint(model, [ηₙ; 1.0; tc_lin - res.value[end]] ∈ MOI.RotatedSecondOrderCone(3))
=#
        @variable(model, ν >= 0)
        @constraint(model, [ν; tc_lin] ∈ MOI.NormOneCone(2))

        @variable(model, L)
        @constraint(model, L == get_cost(δx[:, N]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end]))
        wₘ=1000
        wₙ=50
        wₗ=0.1
        @objective(model, Min, wₘ*μ + r*ηₚ +wₙ*ν + wₗ*ηₗ + L)
        optimize!(model)
        @show objective_value(model)

        est_cost = wₘ*value(μ) + wₙ*value(ν) + value(L) # the linearized cost estimate from the last iterate
        xref_candidate = xref .+ value.(δx)
        uref_candidate = uref .+ value.(δu)
        #@show uref[dil] value.(δu)[dil]
        res_candidate = linearize(xref_candidate, uref_candidate)
        @show value(μ) value(ν) value(L) value(ηₚ) 0.5*sum(value.([reshape(δx, :); reshape(δu, :)])).^2
        push!(delta_lin_hist, res_candidate.derivs[1])
        #@show predicted
        actual = res_candidate.value[1:end-1]
        @show actual[end]
        lin_err = actual .- reshape(xref_candidate[:, 2:N], :)
        #@show lin_err
        actual_cost = wₘ*norm(lin_err, 1) + wₙ*abs(res_candidate.value[end]) + get_cost(reshape(res_candidate.value[1:end-1], nunk, N-1)[:, end])
        push!(costs, est_cost)
        push!(rcosts, actual_cost)
        dk = last_cost - actual_cost
        dl = last_cost - est_cost

        ρᵏ = dk/dl
        @show norm(lin_err, 1) abs(res_candidate.value[end]) get_cost(reshape(res_candidate.value[1:end-1], nunk, N-1)[:, end])
        @show ρᵏ actual_cost est_cost last_cost
        if ρᵏ < ρ₀ # it's gotten worse by too much, increase the penalty by α
            println("REJECT $i $r try $(r*α)")
            r *= α
            #sleep(0.5)
            continue # reject the step
        end
        
        if maximum(abs.(xref .- xref_candidate)) < 1e-6 && maximum(abs.(uref .- uref_candidate)) < 1e-6
            println("DONE < tol")
            push!(uhist, uref)
            push!(xhist, xref)
            push!(whist, value.(w))
            break # done
        end
        res = res_candidate # accept the step
        rd = collect(res.derivs[1]) # res gets clobbered for some reason by linearize?
        rv = collect(res.value)
        xref = xref_candidate
        uref = uref_candidate
        if isinf(last_cost) && est_cost/actual_cost > 2
                println("REJECT (init) $i $r try $(r*α)")
                r *= α
                continue
        end
        if ρᵏ < ρ₁# it's gotten worse by too much (but acceptable), increase the penalty by α
            println("OK $i, CONTRACT $r to $(r*α)")
            r *= α
        elseif ρᵏ < ρ₂
            println("OK $i $r")
            # it's FINE go again 
        else
            println("OK $i, EXPAND $r TO $(r/β)")
            # it hasn't gotten good enough decrease the penalty by a factor of β
            r /= β
        end
        push!(uhist, uref)
        push!(xhist, xref)
        push!(whist, value.(w))
        last_cost = actual_cost
        #sleep(0.5)
    end
    return (uhist, xhist, whist, costs, rcosts, delta_lin_hist, linearize, unknowns(tsys), tunable_parameters(tsys))
end
