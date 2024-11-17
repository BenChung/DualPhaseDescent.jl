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
@component function first_order_hold(; name, N, dt, tmin=0.0, val_pre=0.0, val_post=0.0)
    syms = [Symbol("val$i") for i=1:N]
    #params = [first(@parameters $sym = 0.0, [tunable = true, relevant_time = ((i-1)*dt,i*dt)]) for (i, sym) in enumerate(syms)]
    @parameters vals[1:N]=zeros(N) 
    systems = @named begin
        output = RealOutput()
    end
    eqs = [
        output.u ~ ifelse(t-tmin < 0.0, val_pre, ifelse(t-tmin < dt*N, get_sampled_data_internal(t-tmin, vals, dt, false), ifelse(t-tmin == dt*N, vals[end], val_post)))
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
    convex_mod=nothing, problem_mod=nothing
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
    if !isnothing(problem_mod)
        augmented_system = problem_mod(augmented_system, l, y)
    end
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
    get_cstr = getu(tsys, tsys.y)
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
            terminal_cstr_value = terminal_cstr_fun(inp.u0[:,N], params, last(sim).t) + get_cstr(inp.u0[:,N])
            #@show terminal_cstr_value
            return [reduce(vcat, map(s->s.u[end], sim[1:end-1])); final_state; terminal_cstr_value]
        end
    end
    linearize(inp) = lnz()(inp)


    function sparsity_linearize(states, pars)
        detector = TracerSparsityDetector();
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        res = zeros(nunk * (N-1) + 1);
        Float64.(jacobian_sparsity(lnz(adaptive=false, dt=0.0001, unstable_check=(dt,u,p,t) -> false),linpoint,detector))
    end

    sparsity_pattern = sparsity_linearize(collect(iguess[:, 1:N]), collect(tunable))
    colorvec = matrix_colors(sparsity_pattern)

    dx_ref = zeros(nunk * (N-1) + 1)
    jacfun = (J,x) -> J .= lnz(adaptive=true, dtmax=0.001)(x)
    cache = ForwardColorJacCache(
        jacfun,
        ComponentArray(u0=collect(iguess[:, 1:N]), params=collect(tunable));
        dx = dx_ref,
        colorvec = colorvec,
        sparsity = sparsity_pattern)
    function linearize(states, pars)
#=
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        value = collect(linearize(linpoint))
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        res = DiffResults.JacobianResult(zeros(nunk * (N-1) + 1), linpoint);
        ForwardDiff.jacobian!(res, lnz(adaptive=false, dt=0.0001), linpoint)
=#
        
        linpoint = ComponentArray(u0=collect(states), params=collect(pars))
        jac = zeros(nunk * (N-1) + 1, length(linpoint))
        forwarddiff_color_jacobian!(jac, jacfun, linpoint, cache)
        return (value=ForwardDiff.value(cache), derivs=(jac, ))
        #=
        global value_sparse = 
        global value_sparse_tryme = collect(dx_ref)
        global value_dense = res.value
        global result_sparse = jac
        global result_dense = res.derivs[1]
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
        :convex_mod => convex_mod,
        :pars => params,
        :jac_sparsity => sparsity_pattern,
        :colorvec => colorvec
        ])
end
function do_trajopt(prb; maxsteps=300, sλ=0.05, sX = 0.05, sU = 0.05, ϵ=1e-8, Wmin=1e-2)
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
    params = prb[:pars]


    tic = ModelingToolkit.varmap_to_vars(ic, unknowns(tsys); defaults=Dict([l => 0.0, y => 0.0]))
    xref = collect(iguess)
    uref = collect(tunable)
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

    if !isnothing(convex_mod)
        convex_cstr_fun = convex_mod(tsys)
    else 
        convex_cstr_fun = nothing 
    end

    iref_params,rmk,_ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), params)

    W = ones(nunk * (N-1) + 1)
    λ = zeros(nunk * (N-1) + 1)
    for i=1:maxsteps
        model = Model(Clarabel.Optimizer)
        set_optimizer_attribute(model, "verbose", false)
        @variable(model, δx[1:nunk,1:N])
        @variable(model, w[1:nunk,1:N-1])
        @variable(model, wlin)
        wv = [reshape(w,:); wlin]
        @variable(model, δu[1:nparams])
        @variable(model, tc_lin)
        @constraint(model, [reshape(δx[:, 2:N], :) .+ reshape(w, :); tc_lin + wlin] .== rd * [reshape(δx[:, 1:N], :); δu] .+ rv .- [reshape(xref[:, 2:N], :); 0])
        @constraint(model, reshape(δx[:, 1], :) .== tic .- reshape(xref[:, 1], :))
    
        @variable(model, L)

        objective_expr = L + 
            sum(wv .* W .* wv) + 
            sum(λ .* wv) +
            sum(reshape(δx, :) .* 1/(2*sX) .* reshape(δx, :)) + 
            sum(reshape(δu, :) .* 1/(2*sU) .* reshape(δu, :))
        if !isnothing(convex_cstr_fun)
            symbolic_params = rmk(δu .+ uref)
            objective_expr = convex_cstr_fun(model, δx, xref, symbolic_params, objective_expr)
        end
        
        @constraint(model, tc_lin == 0)

        @constraint(model, L == get_cost(δx[:, N]) + get_cost(reshape(res.value[1:end-1], nunk, N-1)[:, end]))
        @objective(model, Min, objective_expr)
        optimize!(model)

        W .= max.(Wmin, value.(wv) .* W ./ ϵ)
        λ .+= sλ .* value.(wv)
        xref .+= sX .* value.(δx)
        uref .+= sU .* value.(δu)
        res = linearize(xref, uref)

        #@show λ
        @show sum(value.(wv) .* W .* value.(wv))

        push!(delta_lin_hist, res.derivs[1])
        push!(uhist, uref)
        push!(xhist, xref)
        push!(costs, W)
        push!(rcosts, λ)
        push!(whist, value.(w))
        rd = collect(res.derivs[1]) # res gets clobbered for some reason by linearize?
        rv = collect(res.value)
        @show rv[end]
        #sleep(0.5)
    end
    return (uhist, xhist, whist, costs, rcosts, delta_lin_hist, linearize, unknowns(tsys), tunable_parameters(tsys))
end
