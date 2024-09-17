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

struct ParameterUID 
    id::Symbol
end
Symbolics.option_to_metadata_type(::Val{:param_uid}) = ParameterUID
function uid(x, default=nothing)
    p = Symbolics.getparent(x, nothing)
    p === nothing || (x = p)
    return Symbolics.getmetadata(x, ParameterUID, default)
end


struct RelevantTime end
Symbolics.option_to_metadata_type(::Val{:relevant_time}) = RelevantTime
function reltime(x, default = (-Inf,Inf))
    p = Symbolics.getparent(x, nothing)
    p === nothing || (x = p)
    Symbolics.getmetadata(x, RelevantTime, default)
end

Symbolics.@register_symbolic get_sampled_data_internal(t::Float64, buffer::Vector{Float64}, dt, circular_buffer)
function get_sampled_data_internal(a, b, c, d)
    ModelingToolkitStandardLibrary.Blocks.get_sampled_data(a, (b), convert(eltype(b), c), d)
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

@mtkmodel DblInt begin
    @parameters begin
        m, [tunable = false]
        τ = 1.0, [tunable = false]
    end
    @variables begin
        f(t)
        x(t)
        v(t)
    end
    @equations begin
        D(v) ~ τ * f / m
        D(x) ~ τ * v
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

function substitute_parameter_metadata(sys, rewriter)
    @set! sys.systems = map(subsys -> substitute_parameter_metadata(subsys, rewriter), ModelingToolkit.get_systems(sys))
    @set! sys.ps = rewriter(sys, ModelingToolkit.get_ps(sys))
    return sys
end

RuntimeGeneratedFunctions.init(@__MODULE__)
begin
    function CTCS_xform(sys, tspan, running_cost, g, h)
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
        return ODESystem(eqs, t, systems=[sys], name=:augmented_system), l, y
    end

    function augment_offsets(sys)
        csys = complete(sys)
        function rewrite_dervars(sys, varmap; is_root=false)
            diff_vars = filter(v->!any(isequal(v), keys(varmap)), ModelingToolkit.collect_differential_variables(ModelingToolkit.get_eqs(sys)))
            offsets = [let sym = gensym(:ic); (@parameters $sym=0 [param_uid=sym])[1] end for var in diff_vars]
            local_varmap = Dict(diff_vars .=> diff_vars .+ offsets)
            global_varmap = merge(varmap, local_varmap)
            rsys = rewrite_dervars.(ModelingToolkit.get_systems(sys), (global_varmap, ))
            recursive_map = reduce(merge, last.(rsys); init = Dict())
            rns(sym) = is_root ? sym : ModelingToolkit.renamespace(sys, sym)
            output_varmap = [
                rns.(keys(recursive_map)) .=> rns.(values(recursive_map))
                rns.( diff_vars) .=> rns.( offsets)]
            @set! sys.ps = [ModelingToolkit.get_ps(sys); offsets]
            @set! sys.systems = first.(rsys)
            @set! sys.eqs = Symbolics.expand_derivatives.(substitute(ModelingToolkit.get_eqs(sys), global_varmap))
            return sys, Dict(output_varmap)
        end
        mod_sys, vmap = rewrite_dervars(csys, Dict(); is_root=true)
        mod_sys = complete(mod_sys)
        return mod_sys, vmap
    end


    function update_tunables_by_predicate(sys, predicate)
        relevant_mask = [predicate(s) for s in tunable_parameters(sys)]
        relevant = tunable_parameters(sys)[relevant_mask]
        return substitute_parameter_metadata(sys, (sys, ps) -> begin 
            [if all(!isequal(p), relevant)
                setmetadata(p, ModelingToolkit.VariableTunable, false)
            else p end for p in ps] end)
    end

    function update_tunables(sys, relevant_mask)
        relevant = tunable_parameters(sys)[relevant_mask]
        return substitute_parameter_metadata(sys, (sys, ps) -> begin 
            [if all(!isequal(p), relevant)
                setmetadata(p, ModelingToolkit.VariableTunable, false)
            else p end for p in ps] end)
    end

    function build_spbms(sys, N, iguess, given_params, offset_vars)
        base_spbms = ODEProblem[]
        spbms = ODEProblem[]
        relevant_masks = BitSet[]
        update_parms = []
        update_offset = []

        for i=1:N-1
            current_interval = ClosedInterval((i-1)/(N-1), (i)/(N-1))
            relevant_mask = [
                (!isinf(first(reltime(s))) || istunable(s)) && 
                !isempty(intersect(ClosedInterval(reltime(s)...), current_interval)) for s in tunable_parameters(sys)]
            msys = complete(update_tunables(sys, relevant_mask))
            @show tunable_parameters(msys)
            prob = ODEProblem(msys, unknowns(sys) .=> iguess[:, 1], ((i-1)/(N-1), (i)/(N-1)), given_params)
            prob_sense = ODEForwardSensitivityProblem(prob, ForwardSensitivity())
            push!(base_spbms, prob)
            push!(spbms, prob_sense)
            push!(relevant_masks, BitSet([i for i in eachindex(relevant_mask) if relevant_mask[i] == 1]))
            push!(update_parms, setp(msys, tunable_parameters(sys)))
            push!(update_offset, setp(msys, [offset_vars[unk] for unk in unknowns(msys)]))
        end


        return base_spbms, spbms, relevant_masks, update_parms, update_offset
    end

    function build_terminal_linearization(sys, given_params, terminal_cost, Ph)
        terminal_sys = complete(update_tunables_by_predicate(sys, s -> isnothing(uid(s)))) # system with no offsets included
        mparams = ModelingToolkit.MTKParameters(terminal_sys, given_params)
        tunables,repack = SciMLStructures.canonicalize(SciMLStructures.Tunable(), mparams)
        nparams = repack(eachindex(tunables))
        base_ordering = tunable_parameters(terminal_sys)
        tcstr_parmap = Dict(tunable_parameters(terminal_sys) .=> getp(terminal_sys, tunable_parameters(terminal_sys))(nparams)) # maps each parameter to the index at which it appears in the tunable vector
        backmap = eachindex(base_ordering) .=> getindex.((tcstr_parmap, ), base_ordering) # i => j implies parameter i appears at position j in the tunable vector; problem_tunables[last.(backmap)] is in tunables order
        fwdmap = sort(reverse.(backmap), by = first) # j => i implies parameter i appears at position j in the tunable vector; tunables[last.(fwdmap)] is now in problem order
        backarr = last.(backmap)
        
        nstates = length(unknowns(terminal_sys))
        terminal_linearization_point_ref = ComponentArray(
            uf=zeros(nstates),
            pars=tunables)
        terminal_linearization_output_ref = ComponentArray(
            u=zeros(nstates),
            c=zeros(1))
        terminal_linearization_storage = DiffResults.JacobianResult(terminal_linearization_output_ref, terminal_linearization_point_ref)

        terminal_cost_fun = @RuntimeGeneratedFunction(generate_custom_function(terminal_sys, terminal_cost))
        terminal_cstr_fun = @RuntimeGeneratedFunction(generate_custom_function(terminal_sys, Ph))
        get_cost = getu(terminal_sys, terminal_sys.l)
        upd_cost = setu(terminal_sys, terminal_sys.l)
        function terminal_linearization_func(params, tf)
            function compute_terminal_values(output, input)
                _params = SciMLStructures.replace(SciMLStructures.Tunable(), params, input.pars[backarr])
                output.u .= input.uf_linpoint
                upd_cost(output.u, get_cost(input.uf_linpoint) + terminal_cost_fun(input.uf_linpoint, _params, tf))
                output.c .= terminal_cstr_fun(input.uf_linpoint, _params, tf)
            end
        end
        return terminal_linearization_storage, terminal_linearization_func, terminal_sys, mparams
    end

    function build_trajopt_prob(
        sys, tspan, N, given_params, initial_guess,
        ic, running_cost, terminal_cost, 
        g, h, Ph)
        ctcs, l, y = CTCS_xform(sys, tspan, running_cost, g, h)


        (ti, tf) = tspan
        dtime = tf - ti
        tsys = structural_simplify(ctcs)
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
            upd_cost(final_state, get_cost(inp.u0[:,N]) + terminal_cost_fun(inp.u0[:,N], params, last(sim).t))
            terminal_cstr_value = terminal_cstr_fun(inp.u0[:,end], params_, last(sim).t)
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
        reference_linsol = (linearize(xref[:, 1:N],uref))



        offset_system, offset_vars = augment_offsets(ctcs)
        @show unknowns(offset_system)
        simplified_system = structural_simplify(offset_system)
        @show unknowns(simplified_system)
        
        terminal_cost_fun = @RuntimeGeneratedFunction(generate_custom_function(simplified_system, terminal_cost))
        terminal_cstr_fun = @RuntimeGeneratedFunction(generate_custom_function(simplified_system, Ph))

        #upd_start = setp(simplified_system, offset_vars)# have to figure this out from the relevants
        get_cost = getu(simplified_system, simplified_system.l)
        upd_cost = setu(simplified_system, simplified_system.l)
        
        iguess = ModelingToolkit.varmap_to_vars(initial_guess, unknowns(simplified_system); 
            defaults=Dict(unknowns(simplified_system) .=> (zeros(N), )), promotetoconcrete=false)
        iguess = collect(reduce(hcat, iguess)')

        base_spbms, spbms, relmasks, updp, updoffs = build_spbms(simplified_system, N, iguess, given_params, offset_vars)
        
        
        terminal_linearization_storage, terminal_linearization_func, terminal_sys, terminal_params = build_terminal_linearization(simplified_system, given_params, terminal_cost, Ph)
        
        @show unknowns(simplified_system)
        tic = ModelingToolkit.varmap_to_vars(ic, unknowns(simplified_system); defaults=Dict([l => 0.0, y => 0.0]))
        traj_opt_prob = Dict{Symbol, Any}(
            :base_system => simplified_system,
            :terminal_linearization => terminal_linearization_func,
            :terminal_linearization_storage => terminal_linearization_storage,
            :base_subproblems => base_spbms,
            :subproblems => spbms,
            :subproblem_parameter_mask => relmasks,
            :subproblem_parameter_update => updp,
            :subproblem_offset_update => updoffs,
            :ensemble_prob => EnsembleProblem(spbms; safetycopy=false),
            :offset_vars => offset_vars,
            :initial_condition => tic,
            :initial_guess => iguess,
            :given_params => given_params,
            :terminal_sys => terminal_sys,
            :terminal_params => terminal_params,
            :reference_linsol => reference_linsol
        )
        return traj_opt_prob
    end

    map_to_arr(map, len) = getindex.((map, ), 1:len)

    function trajopt(prob)
        xref = prob[:initial_guess]
        uref = ModelingToolkit.varmap_to_vars(prob[:given_params], tunable_parameters(prob[:base_system]); defaults=defaults(prob[:base_system]))
        for (spbm, upd!) in Iterators.zip(prob[:subproblems], prob[:subproblem_parameter_update])
            upd!(spbm, uref)
        end
        for (spbm, upd_u0!, xv) in Iterators.zip(prob[:subproblems], prob[:subproblem_offset_update], eachcol(xref))
            upd_u0!(spbm, xv)
        end
        not_offsets = ModelingToolkit.varmap_to_vars(prob[:given_params], filter(p->isnothing(uid(p)), tunable_parameters(prob[:base_system])); defaults=defaults(prob[:base_system]))

        traj_sensitivities = solve(prob[:ensemble_prob], Tsit5(), EnsembleSerial())
        #return extract_local_sensitivities(traj_sensitivities[1], length(traj_sensitivities[1]))
        final_state, final_jac = extract_local_sensitivities(traj_sensitivities[end], length(traj_sensitivities[end]))

        nstates = length(unknowns(prob[:base_subproblems][end].f.sys))
        input_ref = ComponentArray(
            uf_linpoint=xref[:, end],
            pars=not_offsets)
        output_ref = ComponentArray(
            u=zeros(nstates),
            c=zeros(1))
        ForwardDiff.jacobian!(prob[:terminal_linearization_storage],
            prob[:terminal_linearization](prob[:terminal_params], prob[:base_subproblems][end].tspan[end]),
            output_ref, input_ref)
            @show prob[:terminal_linearization_storage]
            throw("STOP")

        offset_varmap = prob[:offset_vars]
        offset_vars = Set(values(offset_varmap))
        offset_invmap = Dict(values(offset_varmap) .=> keys(offset_varmap))

        jac = prob[:terminal_linearization_storage].derivs[1]
        @show reduce(hcat, final_jac) * zeros(6)
        @show prob[:terminal_linearization_storage].derivs[1]
        @show parameters(prob[:base_system])[collect(prob[:subproblem_parameter_mask][1])]
        @show collect(prob[:subproblem_parameter_mask][1])
        
        base_unknowns = unknowns(prob[:base_system])
        all_params = tunable_parameters(prob[:base_system])
        control_vect = filter(p->all(!isequal(p), offset_vars), all_params)


        N=20
        sense_jac = zeros(nstates*(N-1), nstates * (N-1) + length(all_params))
        for i=1:N-1
            local_spbm = prob[:base_subproblems][i]
            local_unknowns = unknowns(local_spbm.f.sys)
            local_params = tunable_parameters(local_spbm.f.sys)
            mule_params = copy(parameter_values(local_spbm))
            tunables,repack = SciMLStructures.canonicalize(SciMLStructures.Tunable(), mule_params)
            mod_mule = repack(collect(eachindex(local_params)))
            local_params_inorder = local_params[last.(sort(getp(local_spbm.f.sys, local_params)(mod_mule) .=> eachindex(local_params), by=first))]
            local_controls = filter(lp -> any(isequal(lp),control_vect), local_params)

            # offset_map; maps local tunables to global unknowns
            offset_map = [findfirst(isequal(v), local_params_inorder) for v in getindex.((offset_varmap, ), local_unknowns)] 
            # control_map; maps local tunables to global controls
            control_map = [findfirst(isequal(v), local_params_inorder) for v in local_controls]
            # state_map; maps local unknowns to global unknowns
            state_map = [findfirst(isequal(v), local_unknowns) for v in base_unknowns]
            @show state_map


            # offset map and control map cover all local tunables; they tell us what columns of the global jacobian we should write the sensitivities into
            # state map tells us which rows of the global jacobian we should write into
            # the global jacobian is built with inputs δx[1:nunk, 1:N]; δu. The δx is laid out as x[1,1], ..., x[nunk,1], x[2,1] and so on
            # the outputs are x[1:nunk,2:N]
            lref, lsense = extract_local_sensitivities(traj_sensitivities[i], length(traj_sensitivities[i]), true)

            sense_jac[(i-1)*nstates + 1:i*nstates, [(1:nstates) .+ (i - 1) * nstates; [(N-1)*nstates + findfirst(isequal(v), all_params) for v in local_controls]]] .= lsense[state_map, [offset_map; control_map]]
        end
        @show sense_jac

        return sense_jac

    end
    res = trajopt(trajprob)        
    throw("STOP")
    
#=
    function trajopt(
        sys, tspan, N, given_params, initial_guess,
        ic, running_cost, terminal_cost, 
        g, h, Ph
    )
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
    =#

#u,x,wh,ch,rch,dlh,lnz
    trajprob = build_trajopt_prob(sys, (0.0, 1.0), 20, 
        Dict(sys.dblint.m => 0.25), 
        Dict(sys.dblint.x => collect(LinRange(0.0, 1.1, 20)), 
        sys.dblint.v => zeros(20)), 
        [sys.dblint.x => 0.0, sys.dblint.v => 0.0], 
        (sys.dblint.f) .^ 2, 0.0, 
        0.0, 0.0, 
        (30*abs(sys.dblint.v))^4 + (30*abs((sys.dblint.x - 1.0)))^4);
        throw("STOP2")
    trajopt(trajprob)
    2
end
 
using Makie, GLMakie
f = Figure()
ax1 = Makie.Axis(f[1,1])
for xref in x
    lines!(ax1, 1:length(xref[end, :]), xref[end-1, :])
end
ax2 = Makie.Axis(f[2,1])
for xref in x
    lines!(ax2, 1:length(xref[end, :]), xref[end, :])
end
ax3 = Makie.Axis(f[3,1])
for uref in u
    lines!(ax3, 1:length(uref), uref)
end
f
