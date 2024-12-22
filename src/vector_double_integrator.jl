function VecDblInt3(;name)
    pars = @parameters begin
        m, [tunable = false]
        τ = 1.0, [tunable = false]
    end
    vars = ModelingToolkit.@variables begin
        f(t)[1:3]
        x(t)[1:3]
        v(t)[1:3]
    end
    eqs = Symbolics.scalarize.([
        D.(v) .~ τ * f * 2 * 9.81 ./ m - [0.0, -9.81, 0.0]
        D.(x) .~ τ * v
    ])
    return ODESystem(eqs, t, vars, pars; name)
end

function build_example_problem()
    @named dblint = VecDblInt3()
    @named inputx = first_order_hold(N = 20, dt=0.05)
    @named inputy = first_order_hold(N = 20, dt=0.05)
    @named inputz = first_order_hold(N = 20, dt=0.05)
        
    @named model = ODESystem([
        dblint.f[1] ~ inputx.output.u, 
        dblint.f[2] ~ inputy.output.u, 
        dblint.f[3] ~ inputz.output.u], t,
        systems = [dblint, inputx, inputy, inputz])
    return model # structural_simplify(model)
end

sys = build_example_problem()

prb = trajopt(sys, (0.0, 1.0), 20, 
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
    [0.0], 0.0, 
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
