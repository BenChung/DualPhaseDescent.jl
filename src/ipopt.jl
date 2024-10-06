

function ctcs(sys, running_cost, g, h)
    augmenting_vars = ModelingToolkit.@variables begin
        l(t)=0
        y(t)=0
    end
    eqs = [
        D(l) ~ running_cost,
        D(y) ~ sum(max.(0.0, g) .^ 2) + sum(h .^ 2)
    ]
    augmented_system = ODESystem(eqs, t, systems=[sys], name=:augmented_system)
    return augmented_system, augmenting_vars
end
asys, avars = ctcs(sys, sys.dblint.f[1] ^ 2 + sys.dblint.f[2]^2, 0.0, 0.0);

mutable struct PropagateMut{O,S,T}
    N::Int
    tspans::Vector{Tuple{Float64, Float64}}
    pr::O
    upd::S

    last_ip::Matrix{T}
    last_params::Vector{T}
    last_soln::Matrix{T}

    last_dxip::Matrix
    last_dxparams::Vector
    last_dfdx::Matrix
end

function (prop::PropagateMut)(ips, params, x, y)
    return prop(collect(ips), collect(params))[round(Int, x), round(Int, y)]
end
function (prop::PropagateMut)(ips::Vector{Float64}, params)
    ips = reshape(ips, 5, 6)[1:end-1, :]
    if all(prop.last_ip .== ips) && all(prop.last_params .== params)
        #println("hit float")
        return prop.last_soln
    end 
    #println("recompute float")
    for i=1:prop.N-1
        prob_ = prop.upd(prop.pr, ips[i, :], params, prop.tspans[i])
        soln = solve(prob_, Tsit5())
        prop.last_soln[i, :] .= soln.u[end]
    end
    prop.last_ip .= ips
    prop.last_params .= params
    return prop.last_soln
end
function (prop::PropagateMut)(ips::Vector{<:ForwardDiff.Dual}, params)
    ips = reshape(collect(ips), 5, 6)[1:end-1, :]
    if size(ips) == size(prop.last_dxip) && all(prop.last_dxip .== ips) && all(prop.last_dxparams .== params) && typeof(prop.last_dfdx) == typeof(ips)
        #println("hit dual $(typeof(prop.last_dfdx))")
        return prop.last_dfdx
    end 
    #println("recompute dual")
    prop.last_dfdx=similar(ips)
    for i=1:prop.N-1
        prob_ = prop.upd(prop.pr, ips[i, :], params, prop.tspans[i])
        soln = solve(prob_, Tsit5())
        prop.last_dfdx[i, :] .= soln.u[end]
    end
    prop.last_dxip = copy(ips)
    prop.last_dxparams = copy(params)
    #@show (typeof(prop.last_dfdx))
    return prop.last_dfdx
end
Base.show(io::IO, p::PropagateMut) = print(io, "PropagateMut()")
Base.nameof(p::PropagateMut) = :PropagateMut

#@time res = ForwardDiff.jacobian((args) -> prop(args[1:30], args[31:70]), zeros(70))

ssys = structural_simplify(asys)
prob = ODEProblem(ssys, [
    ssys.model.dblint.x => zeros(2), 
    ssys.model.dblint.v => zeros(2), 
    ssys.model.dblint.m => 1.0], (0.0, 1.0))
_, repack = SciMLStructures.canonicalize(SciMLStructures.Tunable(), parameter_values(prob))
function upd(prob, ip, params, tspan) 
    params_ = repack(params)
    return remake(prob, u0=ip, p=params_, tspan=tspan)
end

prop = PropagateMut(5, collect(Iterators.zip(LinRange(0.0, 1.0, 6)[1:end-1], LinRange(0.0, 1.0, 6)[2:end])), 
    prob, upd,
    zeros(4, length(unknowns(ssys))),
    zeros(length(tunable_parameters(ssys))),
    zeros(4, length(unknowns(ssys))),
    zeros(0,0), zeros(0), zeros(0,0)
);

equation_to_jump!(m, eq::Equation) = @constraint(m, eq.lhs == eq.rhs)

function multiple_shooting(sys, N, initial_conditions, terminal_cost, Ph)
    m=Model(Ipopt.Optimizer)

    state_vars = unknowns(sys)
    pars = tunable_parameters(sys)
    JuMP.@variables(m, begin 
        states[1:N, 1:length(state_vars)]
        params[1:length(pars)]
    end)
    replacements = Dict(Symbolics.scalarize(state_vars .=> states[1, :]))
    equation_to_jump!.((m, ), map(ic->substitute(ic, replacements), initial_conditions))
    @operator(m, prop_op, length(states) + length(params) + 2, (args...) -> 
        prop(args[1:length(states)], args[length(states)+1:length(states)+length(params)], args[end-1], args[end]))
    for seg=2:N
        for var=1:length(state_vars)
            @constraint(m, states[seg, var] == GenericNonlinearExpr{VariableRef}(prop_op.head, states..., params..., seg-1, var))
        end
    end
    
    replacements = Dict(Symbolics.scalarize(state_vars .=> states[end, :]))
    for i=1:length(pars)
        @constraint(m, -1.0 <= params[i] <= 1.0)
    end
    @constraint(m, 0 >= substitute(Ph, replacements))
    @objective(m, MIN_SENSE, substitute(terminal_cost, replacements))
    return m, states, params
end
msys, states, params = multiple_shooting(ssys, 5, 
    Symbolics.scalarize([ssys.model.dblint.x .~ 1.0; ssys.model.dblint.v .~ 0.0; ssys.l ~ 0.0; ssys.y ~ 0.0]),
    ssys.l,
    ((sum(abs.(Symbolics.scalarize(ssys.model.dblint.v)))))^2 + (abs(ssys.model.dblint.x[1] - 1) + abs(ssys.model.dblint.x[2] - 1))^2)

set_attribute(msys, "max_cpu_time", 60.0)
optimize!(msys)
value.(states)

@register_array_symbolic (p::PropagateMut)(ip::AbstractMatrix, params::AbstractVector) begin 
    size=let osz = size(ip); (osz[1]-1, osz[2]) end
end
@register_symbolic (p::PropagateMut)(ip::AbstractMatrix, params::AbstractVector, x::Int, y::Int)
function multiple_shooting(sys, N, initial_conditions, terminal_cost, Ph)
    state_vars = unknowns(sys)
    pars = tunable_parameters(sys)
    ModelingToolkit.@variables states[1:N, 1:length(state_vars)] params[1:length(pars)]
    dynamics_eqns = Union{ModelingToolkit.Equation, ModelingToolkit.Inequality}[]
    replacements = Dict(Symbolics.scalarize(state_vars .=> states[1, :]))
    append!(dynamics_eqns, map(ic->substitute(ic, replacements), initial_conditions))
    
    for seg=2:N
        for var=1:length(state_vars)
            push!(dynamics_eqns, states[seg, var] ~ prop(states, params, seg-1, var))
        end
    end

    #append!(dynamics_eqns, reshape(Symbolics.scalarize(states[2:end,:] ~ prop(states, params)), :))
    #append!(dynamics_eqns, reshape(Symbolics.scalarize(states[2:end,:] ~ states[1:end-1,:]), :))
    replacements = Dict(Symbolics.scalarize(state_vars .=> states[end, :]))
    push!(dynamics_eqns, 0 ≲ -substitute(Ph, replacements))
    @show dynamics_eqns substitute(Ph, replacements) substitute(terminal_cost, replacements)
    return OptimizationSystem(
        substitute(terminal_cost, replacements), collect([reshape(states, :); reshape(params, :)]), []; constraints = Symbolics.scalarize.(dynamics_eqns), name=:shooting)
end

msys = multiple_shooting(ssys, 5, 
    Symbolics.scalarize([ssys.model.dblint.x .~ 1.0; ssys.model.dblint.v .~ 0.0; ssys.l ~ 0.0; ssys.y ~ 0.0]),
    ssys.l,
    ((sum(abs.(Symbolics.scalarize(ssys.model.dblint.v)))))^2 + (abs(ssys.model.dblint.x[1] - 1) + abs(ssys.model.dblint.x[2] - 1))^2)
rdy = complete(msys)
prob = OptimizationProblem(rdy, 
    [reshape(rdy.states[:,1:4] .=> 0.0, :); 
    reshape(rdy.states[1:end-1,5:6] .=> 0.0, :); 
    reshape(rdy.states[end,:] .=> [0,0,0,0,0.0,0.0], :); reshape(rdy.params .=> 0.5, :)] |> Symbolics.scalarize, [], 
    grad=true, hess=true, cons_j = true, cons_h = true)
using OptimizationMOI, Ipopt

sol = solve(prob, Ipopt.Optimizer())
