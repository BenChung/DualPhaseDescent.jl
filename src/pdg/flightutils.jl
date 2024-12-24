
control_invmap = CSV.read("src/kerbal/control_invmap.csv", DataFrame)
kts = (sort(unique(control_invmap[:, :mach])), sort(unique(control_invmap[:, :aoa1])), sort(unique(control_invmap[:, :aoa2])))

function buildlut(field)
    lut = zeros(length.(kts))
    for r in eachrow(control_invmap)
        ind = (findfirst(==(r.mach), kts[1]), findfirst(==(r.aoa1), kts[2]), findfirst(==(r.aoa2), kts[3]))
        lut[ind...] = r[field]
    end
    return extrapolate(interpolate(kts, lut, Gridded(Linear())), Flat())
end

# act_x = lin_x * cmd[2] + offs_x, act_z = lin_z * cmd[1] + offs_z
linx = buildlut(:lin_x)
linz = buildlut(:lin_z)
offsx = buildlut(:offs_x)
offsz = buildlut(:offs_z)
@register_array_symbolic compute_deflection_feedforward(aoa1, aoa2, mach, u::AbstractArray) begin 
    size = (3,)
    eltype = Float64
end
function compute_deflection_feedforward(aoa1, aoa2, mach, u)
    return (linx(aoa1, aoa2, mach) * u[2] + offsx(aoa1, aoa2, mach), 0.0, linz(aoa1, aoa2, mach) * u[2] + offsz(aoa1, aoa2, mach))
end

function build_ff_sys()
    ModelingToolkit.@variables finvec(t)[1:3]
    @named ffed = ODESystem([Symbolics.scalarize(finvec ~ compute_deflection_feedforward(probsys.veh.alpha1, probsys.veh.alpha2, probsys.veh.mach, probsys.veh.ua))], t)
    return compose(ffed, probsys)
end

flightsys = structural_simplify(build_ff_sys())


soln = propagate_sol(ssys, u)
Makie.lines!(soln.t, getu(ssys, ssys.veh.ua[2])(soln))
Makie.lines(soln.t, getu(ssys, compute_deflection_feedforward(ssys.veh.alpha1, ssys.veh.alpha2, ssys.veh.mach,Symbolics.scalarize(ssys.veh.ua))[1])(soln))
Makie.lines!(soln.t, getu(ssys, compute_deflection_feedforward(ssys.veh.alpha1, ssys.veh.alpha2, ssys.veh.mach,Symbolics.scalarize(ssys.veh.ua))[3])(soln))