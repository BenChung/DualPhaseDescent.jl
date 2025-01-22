
control_invmap = CSV.read("src/kerbal/control_invmap.csv", DataFrame)

struct ControlMapInterp{T1, T2, T3}
    upper_lim_lut::T1
    lower_lim_lut::T2
    invmap_lut::T3
end
@register_symbolic (c::ControlMapInterp)(mach, aoa1, aoa2, demand)
function (c::ControlMapInterp)(mach, aoa1, aoa2, demand)
    ll,ul = c.lower_lim_lut(mach, aoa1, aoa2), c.upper_lim_lut(mach, aoa1, aoa2)
    c.invmap_lut(mach, aoa1, aoa2, 2*(demand + ll)/(ul+ll)-1)
end

function make_inv_ctrl_lookup(invmap, axis)
    dat = invmap |> filter(:axis => ==(string(axis)))
    mach_kts = unique(dat[:, :mach]); mach_map = Dict(mach_kts .=> eachindex(mach_kts))
    aoa1_kts = unique(dat[:, :aoa1]); aoa1_map = Dict(aoa1_kts .=> eachindex(aoa1_kts))
    aoa2_kts = unique(dat[:, :aoa2]); aoa2_map = Dict(aoa2_kts .=> eachindex(aoa2_kts))
    demand_kts = unique(dat[:, :demand]); demand_map = Dict(demand_kts .=> eachindex(demand_kts))
    lut = zeros((length(mach_kts), length(aoa1_kts), length(aoa2_kts), length(demand_kts)))
    for row in eachrow(dat)
        mind = mach_map[row[:mach]]
        a1ind = aoa1_map[row[:aoa1]]
        a2ind = aoa2_map[row[:aoa2]]
        dind = demand_map[row[:demand]]
        lut[mind, a1ind, a2ind, dind] = row[:control]
    end
    itp = Interpolations.extrapolate(Interpolations.interpolate((mach_kts, aoa1_kts, aoa2_kts, demand_kts), lut, Interpolations.Gridded(Interpolations.Linear())), Interpolations.Flat())
    if axis == :actx
        return ControlMapInterp(upper_lim2_lut, lower_lim2_lut, itp)
    elseif axis == :actz 
        return ControlMapInterp(upper_lim1_lut, lower_lim1_lut, itp)
    end
end

actx_lookup = make_inv_ctrl_lookup(control_invmap, :actx);
actz_lookup = make_inv_ctrl_lookup(control_invmap, :actz);

function build_ff_sys()
    ModelingToolkit.@variables finvec(t)[1:3] tphys(t)=0
    aoa1 = probsys.veh.alpha1
    aoa2 = probsys.veh.alpha2
    mach = probsys.veh.mach
    eqs = [
        finvec[1] ~ actx_lookup(mach, aoa1, aoa2, probsys.veh.ua[2]), # + offsx(aoa1, aoa2, mach),
        finvec[2] ~ 0.0,
        finvec[3] ~ actz_lookup(mach, aoa1, aoa2, probsys.veh.ua[1]), #linz(aoa1, aoa2, mach) * probsys.veh.ua[1] + offsz(aoa1, aoa2, mach),
        D(tphys) ~ probsys.veh.τc]
    @named ffed = ODESystem(eqs, t)
    return compose(ffed, probsys)
end

flightsys = structural_simplify(build_ff_sys())

#=

soln = propagate_sol(flightsys, u; veh=flightsys.model.veh)

traj_df = DataFrame(map(x->x => [], [
    :t, :posx, :posy, :posz, :vx, :vy, :vz, :Rx, :Ry, :ox, :oy, :finx, :finy, :finz, :thx, :thy, :thz,
    :aero_x, :aero_y, :aero_z, :at_x, :at_y, :at_z
]))
for row in soln(LinRange(0.0, 1.0, 2000), idxs=Symbolics.scalarize.([
    flightsys.tphys; 
    flightsys.model.veh.posp;
    flightsys.model.veh.vp;
    flightsys.model.veh.Rp;
    flightsys.model.veh.ω;
    flightsys.finvec; 
    flightsys.model.veh.u;
    flightsys.model.veh.aero_force;
    Symbolics.scalarize(flightsys.model.veh.ρᵣ * (flightsys.model.veh.ctrl_torque .+ flightsys.model.veh.body_torque))]))
    push!(traj_df, row)
end
CSV.write("traj.csv", traj_df)

Makie.lines!(soln.t, getu(ssys, ssys.veh.ua[2])(soln))
Makie.lines(soln.t, getu(ssys, compute_deflection_feedforward(ssys.veh.alpha1, ssys.veh.alpha2, ssys.veh.mach,Symbolics.scalarize(ssys.veh.ua))[1])(soln))
Makie.lines!(soln.t, getu(ssys, compute_deflection_feedforward(ssys.veh.alpha1, ssys.veh.alpha2, ssys.veh.mach,Symbolics.scalarize(ssys.veh.ua))[3])(soln))
=#