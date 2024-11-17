
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
ssys = structural_simplify(sys)


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
prb = trajopt(sys, (0.0, 1.0), 20, 
    Dict(sys.dblint.m => 0.25), 
    Dict(sys.dblint.x => collect(LinRange(0.0, 1.0, 20)), 
    sys.dblint.v => zeros(20)), 
    [sys.dblint.x => 0.0, sys.dblint.v => 0.0], 
    (sys.dblint.f) .^ 2, 0.0, 
    0.0, 0.0, 
    100*(sys.dblint.v)^2 + 100*(sys.dblint.x - 1.0)^2, 
    (tsys) -> begin 
        get_inp = getp(tsys, tsys.model.input.vals)
        return function (model, δx, xref, symbolic_params, objexp)
            ctrl = get_inp(symbolic_params)
            @constraint(model, ctrl .<= 1.0)
            @constraint(model, ctrl .>= -1.0)
            return objexp
        end
    end);

ui,xi,wi,Wh,lh,_,_,unk,_ = do_trajopt(prb; maxsteps=10, sλ=0.05,sX=0.1,sU=0.1,Wmin=100000,ϵ=1e-9); findmax(abs.(wi[end]))
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
