using Rotations
import Quaternions

function PoweredDescent(;name)
    pars = @parameters begin
        m, [tunable = false]
        τ = 1.0, [tunable = false]
        eng_r[1:3] = [0.0,-1.0,0.0], [tunable = false]
        MoI[1:3,1:3] = I(3), [tunable = false]
        MoI_inv[1:3, 1:3] = I(3), [tunable = false]
    end
    vars = ModelingToolkit.@variables begin
        qbi(t)[1:4]
        fᵢ(t)[1:3]
        fₗ(t)[1:3]
        x(t)[1:3]
        v(t)[1:3]
        ω(t)[1:3]
        trq(t)[1:3]
    end
    eqs = Symbolics.scalarize.([
        D.(v) ./ τ .~ fᵢ ./ m - [0.0, 9.81, 0.0]
        D.(x) ./ τ .~ v
        D.(ω) ./ τ .~ MoI_inv * (trq + cross(MoI * ω, ω))
        D.(qbi) ./ τ .~ Rotations.kinematics(QuatRotation(Quaternions.Quaternion(qbi...), false), Symbolics.scalarize(ω))
        fᵢ .~ QuatRotation(Quaternions.Quaternion(qbi...), false) * Symbolics.scalarize(fₗ)
        trq .~ cross(eng_r, fₗ)
    ])
    return ODESystem(eqs, t, vars, pars; name)
end

function build_example_problem()
    @named dblint = PoweredDescent()
    @named inputx = first_order_hold(N = 20, dt=0.05)
    @named inputy = first_order_hold(N = 20, dt=0.05)
    @named inputz = first_order_hold(N = 20, dt=0.05)
        
    @named model = ODESystem([
        dblint.fₗ[1] ~ inputx.output.u, 
        dblint.fₗ[2] ~ inputy.output.u, 
        dblint.fₗ[3] ~ inputz.output.u], t,
        systems = [dblint, inputx, inputy, inputz])
    return model # structural_simplify(model)
end

sys = build_example_problem()

import RuntimeGeneratedFunctions
function (f::RuntimeGeneratedFunctions.RuntimeGeneratedFunction{argnames, cache_tag, context_tag, id})(args::Vararg{Any, N}) where {N, argnames, cache_tag, context_tag, id}
    try
        RuntimeGeneratedFunctions.generated_callfunc(f, args...)
    catch e 
        @error "Caught error in RuntimeGeneratedFunction; source code follows"
        func_expr = Expr(:->, Expr(:tuple, argnames...), RuntimeGeneratedFunctions._lookup_body(cache_tag, id))
        @show func_expr
        rethrow(e)
    end
end
u,x,wh,ch,rch,dlh,lnz = trajopt(sys, (0.0, 1.0), 20, 
    Dict(sys.dblint.m => 1.0, sys.dblint.τ => 1.0), 
    Dict(
        sys.dblint.x[1] => collect(LinRange(0.0, 1.0, 20)), 
        sys.dblint.x[2] => collect(LinRange(0.0, 1.0, 20)),
        sys.dblint.x[3] => collect(LinRange(0.0, 1.0, 20)),
        sys.dblint.v[1] => zeros(20),
        sys.dblint.v[2] => zeros(20),
        sys.dblint.v[3] => zeros(20)), 
    [sys.dblint.x[1] => 0.0, 
     sys.dblint.v[1] => 0.0,
     sys.dblint.x[2] => 0.0, 
     sys.dblint.v[2] => 0.0,
     sys.dblint.x[3] => 0.0, 
     sys.dblint.v[3] => 0.0], 
    sum([sys.dblint.f[1], sys.dblint.f[2], sys.dblint.f[3]].^2), 0.0, 
    0.0, 0.0, 
    ((sum((sys.dblint.v).^2))) + ((sum((sys.dblint.x .- 1) .^2))));

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
