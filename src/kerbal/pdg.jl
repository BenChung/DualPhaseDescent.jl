
using Revise, GLMakie, Makie
using ModelingToolkit, ECOS, GeometryBasics, OrdinaryDiffEq
using LinearAlgebra, Rotations, Statistics, Distributions
using SCPToolbox, SCPModelingToolkit, LazySets
using Polyhedra, QHull, Delaunay, ConvexBodyProximityQueries
using Quaternions


include("aero_processing.jl")


rquat(R) = QuatRotation(R[1], R[2], R[3], R[4], false)
iquat(R) = QuatRotation(R[1], -R[2], -R[3], -R[4], false)
Base.:/(q::Quaternions.Quaternion, x::Num) = Quaternions.Quaternion(q.s / x, q.v1 / x, q.v2 / x, q.v3 / x)

# we assume a constant density atmosphere
# compute basic aerodynamic parameters
temperature = 293 # K
mach = 331 * sqrt(temperature/273)
struct Pitching{T} Cm::T end
@register_symbolic (p::Pitching)(mach::Num, aoa::Num)
function (p::Pitching)(mach, aoa) 
    return p.Cm(mach, aoa)
end
Base.nameof(::Pitching) = :Pitching

struct Thrusting{T} Cx::T end
@register_symbolic (t::Thrusting)(mach::Num, aoa::Num)
function (t::Thrusting)(mach, aoa) 
    res = t.Cx(mach, aoa)
    return res
end
Base.nameof(::Thrusting) = :Thrusting

struct Lifting{T} Cn::T end
@register_symbolic (l::Lifting)(mach::Num, aoa::Num)
function (l::Lifting)(mach, aoa) # u"N"
    return l.Cn(mach, aoa)
end
Base.nameof(::Lifting) = :Lifting

@register_symbolic printfun(num::Num)
printfun(l) = begin @show l; l end

@register_symbolic sqrt_smooth(a::Num)
function sqrt_smooth(s)
    return ifelse(s == 0.0, s, sqrt(s))
end

@register_symbolic atand_smooth(a::Num, b::Num)
function atand_smooth(a,b)
    return atand(a,b)
end

function angle(point1, point2) 
    return atand_smooth(sqrt_smooth(sum(cross(point1, point2) .^ 2)), dot(point1, point2))
end

function angle_in_plane(vec, ref, normal)
    return atand_smooth(dot(cross(vec, ref), normal), dot(vec, ref))
end

drag_model = make_interpolation(:d)
lift_model = make_interpolation_adjusted(:lx)
moment_model = make_interpolation_adjusted(:tz)
struct FinForce{T}
    lut::T
end
@register_symbolic (f::FinForce)(m::Num,a1::Num,a2::Num,d::Num, i)
(f::FinForce)(in::Vector{Float64}) = f(in[1], in[2], in[3], in[4])
(f::FinForce)(m::Real,a1::Real,a2::Real,d::Real, i) = f.lut(m,a1,a2,d)[[3,1,2]][i]
Base.nameof(::FinForce) = :FinForce

f1 = FinForce(make_lut_for("f1"));
f2 = FinForce(make_lut_for("f2"));
f3 = FinForce(make_lut_for("f3"));
f4 = FinForce(make_lut_for("f4"));

finlocs = [1.4682299836624186 9.188123202888175 -1.0078209977670838e-5;
    -7.716612765246156e-6 9.162568775377267 1.4674932738649753;
    -1.4615854536794377 9.179147545041362 -1.1251217936880664e-5;
    -8.076746417132186e-6 9.186309310308038 -1.4664886069968053][:,[3,1,2]]

function make_vehicle(Cx, Cn, Cm; 
    iJz_wet=[1/30000.19, 1/30000.19, 1/100],
    iJz_dry=[1/30000.19, 1/30000.19, 1/100],
    mdry=1000,
    fuel_mass=1000,
    engine_offset = [0, 0, -8], name)
    @parameters iJz_delta[1:3] = iJz_wet - iJz_dry#.*u"kg" .* u"m^2")
    @parameters iJz_dry[1:3] = iJz_dry#.*u"kg" .* u"m^2")
    @parameters wind[1:3] = [0,0,0]
    @parameters Cmdot = 4.0
    @parameters engine_offset[1:3] = engine_offset
    @parameters fin_offset[1:3] = [0, 0, 1]
    @parameters ISP = 300 fuel_mass = fuel_mass mdry = mdry
    @parameters speed_of_sound = 340 ρ=1.225
    @parameters fin_locs[1:4,1:3] = finlocs

    @variables t
    D = Differential(t)
    @variables pos(t)[1:3] [dynamics=true] v(t)[1:3] [dynamics=true] m(t) [dynamics=true] propellant_fraction(t)
    @variables R(t)[1:4] [dynamics=true] ω(t)[1:3] [dynamics=true] th(t)[1:3] mach(t)

    @variables free_dynamic_pressure(t) phi(t) vel(t) iJz(t)[1:3] alpha1(t) alpha2(t)
    @variables aero_moment(t) aero_torque(t)[1:3] torque(t)[1:3] aero_force(t)[1:3] Cdfs(t) Clfs(t) net_torque(t)[1:3]
    @variables lift_dir(t)[1:3] local_wind_vec(t)[1:3] alpha(t) accel(t)[1:3]
    @variables fin_force(t)[1:4, 1:3] fin1_force(t)[1:3] fin2_force(t)[1:3] fin3_force(t)[1:3] fin4_force(t)[1:3] τc(t)
    @variables u(t)[1:3] [scpcontrol = true] delta(t)[1:4] [scpcontrol=true]

    @parameters τ1 = 1.0 [tunable = true] τ2 = 1.0 [tunable = true]
    
    eqns = [
        local_wind_vec .~ rquat(R) * Symbolics.scalarize(v+ wind);
        alpha ~ angle(v+ wind, rquat(R) * [0,0,-1]);
        alpha1 ~ angle_in_plane(local_wind_vec, [0,0,-1], [0,1,0]);
        alpha2 ~ angle_in_plane(local_wind_vec, [0,0,-1], [1,0,0]);
        mach ~ norm(v+wind)/speed_of_sound;
        Cdfs ~ 1000*(Thrusting(Cx))(mach, alpha);
        Clfs ~ -1000*(Lifting(Cn))(mach, alpha);
        aero_moment ~ -1000*(Pitching(Cm))(mach, alpha) * ρ * norm(v+wind);
        fin_force[1, :] .~ f1.(mach,alpha1,alpha2,delta[1],1:3);
        fin_force[2, :] .~ f2.(mach,alpha1,alpha2,delta[2],1:3);
        fin_force[3, :] .~ f3.(mach,alpha1,alpha2,delta[3],1:3);
        fin_force[4, :] .~ f4.(mach,alpha1,alpha2,delta[4],1:3);
        aero_force .~ 
            (Cdfs * ρ * norm(v+ wind)) .* (v+ wind) .+ 
            Clfs * ρ * collect(cross(Symbolics.scalarize(v+wind), cross(iquat(R) * [0,0,-1],Symbolics.scalarize(v+wind)))) .+ 
            iquat(R) * Symbolics.scalarize(sum(fin_force, dims=1)');
        aero_torque .~ Symbolics.scalarize(cross([0,0,-1], rquat(R) * Symbolics.scalarize(v + wind)) .* aero_moment) .+
                        iquat(R) *sum(cross(fin_locs[i, :], fin_force[i, :]) for i=1:4)# .+ cross(fin_offset, [actuator_normal; actuator_axial]));
        net_torque .~ cross(engine_offset, Symbolics.scalarize(u * m *mdry/9.82)) + aero_torque;
        τc ~ ifelse(t <= 0.5, τ1, τ2);

        iJz .~ iJz_dry # + iJz_delta * propellant_fraction;
        propellant_fraction ~ (m*mdry - mdry)/(fuel_mass);
        D(m) ~ -τc*norm(u)*(m*mdry)/(9.82 * ISP*mdry);

        D.(ω) .~ -τc.*collect(Symbolics.scalarize(Symbolics.scalarize(iquat(R) * Symbolics.scalarize(iJz .* net_torque))));
        D.(R) .~ τc.*Rotations.kinematics(rquat(R), Symbolics.scalarize(ω));

        Symbolics.scalarize(D.(v) .~ τc.*([0, 0, -9.8] .+ (aero_force) ./ (m*mdry) .+ iquat(R) * Symbolics.scalarize(u)));
        Symbolics.scalarize(D.(pos) .~ τc.*v);
    ]
    return ODESystem(Symbolics.scalarize.(eqns), t; name = name)
end
@named vehicle = make_vehicle(drag_model, lift_model, moment_model;
    mdry=mdry,
    fuel_mass=fuel_mass, 
    iJz_dry=[1/xformed_tensors[end][1,1], 1/xformed_tensors[end][3,3], 1/xformed_tensors[end][2,2]],
    iJz_wet=[1/xformed_tensors[1][1,1], 1/xformed_tensors[1][3,3], 1/xformed_tensors[1][2,2]],
    engine_offset=engine_position[[1,3,2]])

sim,_ = structural_simplify(vehicle, (SCPModelingToolkit.control_variables(vehicle), SCPModelingToolkit.dynamics_variables(vehicle)));

tf_max = 15.0
tf_min = 0.25
pos_init = [50.0,50.0,600.0]
vel_init = [0,0,-100.0]
R_init = [1.0,0,0,0]
ω_init = [0,0,0.0]
m_init = (mdry + fuel_mass*0.1)/mdry

pos_final = [0,0,0.0]
vel_final = [0,0,0.0]
R_final = [1.0,0,0,0]
ω_final = [0,0,0.0]

delta_max = 25.0

prob = SCPModelingToolkit.SCPtProblem(
scale_advice = Dict(
    sim.τ1 => (tf_min, (tf_min + 1.0 * (tf_max - tf_min))),
    sim.τ2 => (tf_min, (tf_min + 1.0 * (tf_max - tf_min)))
    ),
dynamics = sim,
constraints = SCPModelingToolkit.ConicConstraint[
    SCPModelingToolkit.ConicConstraint("max_duration2", sim.τ2 - tf_max, NONPOS),
    SCPModelingToolkit.ConicConstraint("min_duration2", tf_min - sim.τ2, NONPOS),
    SCPModelingToolkit.ConicConstraint("max_duration", sim.τ1 - tf_max, NONPOS),
    SCPModelingToolkit.ConicConstraint("min_duration", tf_min - sim.τ1, NONPOS),
    #SCPModelingToolkit.ConicConstraint("min_duration", tf_min - sim.τ2, NONPOS),
    SCPModelingToolkit.ConicConstraint("max_f1", sim.delta[1] - delta_max, NONPOS),
    SCPModelingToolkit.ConicConstraint("min_f1", -delta_max - sim.delta[1], NONPOS),
    SCPModelingToolkit.ConicConstraint("max_f2", sim.delta[2] - delta_max, NONPOS),
    SCPModelingToolkit.ConicConstraint("min_f2", -delta_max - sim.delta[2], NONPOS),
    SCPModelingToolkit.ConicConstraint("max_f3", sim.delta[3] - delta_max, NONPOS),
    SCPModelingToolkit.ConicConstraint("min_f3", -delta_max - sim.delta[3], NONPOS),
    SCPModelingToolkit.ConicConstraint("max_f4", sim.delta[4] - delta_max, NONPOS),
    SCPModelingToolkit.ConicConstraint("min_f4", -delta_max - sim.delta[4], NONPOS),
    SCPModelingToolkit.ConicConstraint("thrust_direction", [sim.u[3], sim.u[1], sim.u[2]], SOC),
    SCPModelingToolkit.ConicConstraint("max_thrust", (t,k) -> [Num(t > 0.5 ? 40.0 : 0.1); sim.u], SOC),
    SCPModelingToolkit.ConicConstraint("quaternion_sane", [Num(2.0); sim.R], SOC),
    SCPModelingToolkit.ConicConstraint("glideslope", [sim.pos[3] * tand(50.0), sim.pos[1], sim.pos[2]], SOC),
    SCPModelingToolkit.ConicConstraint("terminal_velocity", (t,k) -> [Num(k >= 25 ? 0.2 : 500.0), sim.v[1], sim.v[2]], SOC),

    #SCPModelingToolkit.ConicConstraint("non_violation", 1000*cstr - 0.01   , NONPOS)
],
terminal_cost = 0.0,
running_cost = (t,k) -> sim.u[1]^2 + sim.u[2]^2 + sim.u[3]^2,
nonconvex_constraints = Num[
    -sim.pos[3],
    #5 - sqrt(sim.u[1]^2 + sim.u[2]^2 + sim.u[3]^2)
    ],
bcs = Dict{Symbol, Vector{Num}}(
    :ic => [sim.pos;sim.v;sim.R;sim.ω;sim.m] - [pos_init; vel_init; R_init; ω_init; m_init],
    :tc => [sim.pos;sim.v;sim.R;sim.ω] - [pos_final; vel_final; R_final; ω_final],
),
initalizer = SCPModelingToolkit.StraightlineInterpolate(
    Dict{Num, Float64}(Symbolics.scalarize.([
        sim.pos .=> pos_init; 
        sim.v .=> vel_init; 
        sim.R .=> R_init; 
        sim.ω .=> ω_init; 
        sim.u .=> [0.0,0.0,9.81];
        sim.delta .=> 0.0; 
        sim.m => m_init])), 
        Dict{Num, Float64}(Symbolics.scalarize.([
            sim.pos .=> pos_final; 
            sim.v .=> vel_final; 
            sim.R .=> R_final; 
            sim.ω .=> ω_final; 
            sim.u .=> [0.0,0.0,9.81]; 
            sim.delta .=> 0.0;
            sim.m => m_init/2])),
    Dict{Num, Float64}([
        sim.τ1 => 0.5 * (tf_min + tf_max)/2,
        sim.τ2 => 0.5 * (tf_min + tf_max)/2
        ])),
callback = spbm->begin  false end
);

soln = SCPModelingToolkit.solve!(prob, ECOS; 
use_forwarddiff=true, N=30, iter_max=50, solver_options = Dict("verbose" => false),
wtr = 0.25);

@named controlled = ODESystem(Symbolics.scalarize(vehicle.u .~ [0,0,5000000]),vehicle.iv;systems=[vehicle])
sim = structural_simplify(controlled)
prob = ODEProblem(sim, Symbolics.scalarize.([
    sim.vehicle.v .=> [0.0,0,-500];
    sim.vehicle.pos .=> [0.0,0,0];
    sim.vehicle.ω .=> [0.0,0,0];
    sim.vehicle.R .=> [1.0,0,0,0];
    sim.vehicle.m => mdry + fuel_mass
]), (0.0, 12.0),
[sim.vehicle.Cmdot => 1.0]
)
@time sol = solve(prob, Tsit5(); dtmax=0.1)

norm(soln[2].subproblems[end].sol.vd, Inf)

thr_max = maximum(norm.(eachcol(soln[1].ud[[7,5,6],:])))

soln_back = soln 

td2, td1 = soln[1].p
switchover = findfirst(t -> t > 0.5, soln[1].td)
aero_phase = map(t -> t <= soln[1].td[switchover], soln[1].td)
powered_phase = map(t -> (t >= soln[1].td[switchover]), soln[1].td)
tact = map(t->ifelse(t <= 0.5, t, t - 0.5), soln[1].td) .* map(t -> ifelse(t <= 0.5, td1, td2), soln[1].td) .+ 
    map(t -> ifelse(t <= 0.5, 0.0, td1 * 0.5), soln[1].td)

    #GLMakie.activate!()
    import CairoMakie 
    CairoMakie.activate!()


f=Figure(size=(1400,900))
a=Axis3(f[1:3,1], aspect=:data, azimuth = -1.275π, xticks=0:20:60, yticks=0:20:60)
Makie.lines!(a,soln[1].xd[end-2,:], soln[1].xd[end-1,:], soln[1].xd[end,:])
Makie.arrows!(a,soln[1].xd[end-2,15:end], soln[1].xd[end-1,15:end], soln[1].xd[end,15:end], 
    eachrow(reduce(hcat, map((r,u)->iquat(r) * u, eachrow(soln[1].xd[5:8,15:end]'), eachrow(10 * (soln[1].ud[[7,5,6],15:end]') ./ thr_max ))))...;
    arrowsize=2.5)
Makie.arrows!(a,soln[1].xd[end-2,:], soln[1].xd[end-1,:], soln[1].xd[end,:], 
    eachrow(reduce(hcat, map((r)->iquat(r) * [0,0,1], eachrow(soln[1].xd[5:8,:]'))))...)
for i=1:size(soln[1].xd)[2]
    bv1 = soln[1].xd[end-2:end,i] + iquat(soln[1].xd[5:8,i]) * [0,0,10]
    bv2 = soln[1].xd[end-2:end,i] + iquat(soln[1].xd[5:8,i]) * [0,10,0]
    bv3 = soln[1].xd[end-2:end,i] + iquat(soln[1].xd[5:8,i]) * [10,0,0]
    Makie.lines!(a,
        [soln[1].xd[end-2,i], bv1[1]], 
        [soln[1].xd[end-1,i], bv1[2]], 
        [soln[1].xd[end,i], bv1[3]], color=:red)
    Makie.lines!(a,
        [soln[1].xd[end-2,i], bv2[1]], 
        [soln[1].xd[end-1,i], bv2[2]], 
        [soln[1].xd[end,i], bv2[3]], color=:blue)
    Makie.lines!(a,
        [soln[1].xd[end-2,i], bv3[1]], 
        [soln[1].xd[end-1,i], bv3[2]], 
        [soln[1].xd[end,i], bv3[3]], color=:green)
end
b1 = Axis(f[1, 2], title="Position")
b2 = Axis(f[1, 2], yaxisposition = :right)
zm = Makie.lines!(b1, tact[powered_phase], soln[1].xd[end, powered_phase], label="z(m)", color=:red)
xm = Makie.lines!(b2, tact[powered_phase], soln[1].xd[end-2, powered_phase], label="x(m)", color=:green)
ym = Makie.lines!(b2, tact[powered_phase], soln[1].xd[end-1, powered_phase], label="y(m)", color=:blue)
zam = Makie.lines!(b1, tact[aero_phase], soln[1].xd[end, aero_phase], label="z(m)", color=:red, linestyle=:dot)
xam = Makie.lines!(b2, tact[aero_phase], soln[1].xd[end-2, aero_phase], label="x(m)", color=:green, linestyle=:dot)
yam = Makie.lines!(b2, tact[aero_phase], soln[1].xd[end-1, aero_phase], label="y(m)", color=:blue, linestyle=:dot)
Legend(f[1,2], [xm, ym, zm], ["x(m)", "y(m)", "z(m)"], "Axis",
    tellheight = false,
    tellwidth = false,
    margin = (5,5,5,5),
    halign = :left, 
    valign = :bottom)
vi = [9, 10, 11]
b3 = Axis(f[2, 2], title="Z Velocity (m/s)")
b4 = Axis(f[3, 2], title="X-Y Velocity (m/s)", xlabel="Time (s)")
zvm = Makie.lines!(b3, tact[powered_phase], soln[1].xd[vi[3], powered_phase], color=:red)
xvm = Makie.lines!(b4, tact[powered_phase], soln[1].xd[vi[1], powered_phase], label="x", color=:green)
yvm = Makie.lines!(b4, tact[powered_phase], soln[1].xd[vi[2], powered_phase], label="y", color=:blue)
zvam = Makie.lines!(b3, tact[aero_phase], soln[1].xd[vi[3], aero_phase], color=:red, linestyle=:dot)
xvm = Makie.lines!(b4, tact[aero_phase], soln[1].xd[vi[1], aero_phase], label="x", color=:green, linestyle=:dot)
yvm = Makie.lines!(b4, tact[aero_phase], soln[1].xd[vi[2], aero_phase], label="y", color=:blue, linestyle=:dot)
Legend(f[2,2], [zvam, zvm], ["aerodynamic", "powered"], "Phase",
    tellheight = false,
    tellwidth = false,
    margin = (5,5,5,5),
    halign = :left, 
    valign = :top)
b5 = Axis(f[1, 3], title="AoA (°)")
Makie.lines!(b5, tact[powered_phase], angle.(map(r->QuatRotation(r...) * [0,0,1], eachrow(soln[1].xd[5:8,powered_phase]')), ([0,0,1],)), color="#56B4E9")
Makie.lines!(b5, tact[aero_phase], angle.(map(r->QuatRotation(r...) * [0,0,1], eachrow(soln[1].xd[5:8,aero_phase]')), ([0,0,1],)), color="#56B4E9",linestyle=:dot)
b6 = Axis(f[2, 3], title="Omega (°/s)")
omega_i = [2,3,4]
zvm = Makie.lines!(b6, tact[powered_phase], -soln[1].xd[omega_i[3], powered_phase], color=:red)
xvm = Makie.lines!(b6, tact[powered_phase], -soln[1].xd[omega_i[1], powered_phase], label="x", color=:green)
yvm = Makie.lines!(b6, tact[powered_phase], -soln[1].xd[omega_i[2], powered_phase], label="y", color=:blue)
zvm = Makie.lines!(b6, tact[aero_phase], -soln[1].xd[omega_i[3], aero_phase], color=:red, linestyle=:dot)
xvm = Makie.lines!(b6, tact[aero_phase], -soln[1].xd[omega_i[1], aero_phase], label="x", color=:green, linestyle=:dot)
yvm = Makie.lines!(b6, tact[aero_phase], -soln[1].xd[omega_i[2], aero_phase], label="y", color=:blue, linestyle=:dot)
b7 = Axis(f[3, 4], title="Fin position (°)", xlabel="Time (s)")
act_i = [1,2,3,4]
zvm = Makie.lines!(b7, tact[powered_phase], soln[1].ud[act_i[1], powered_phase], color=:red)
xvm = Makie.lines!(b7, tact[powered_phase], soln[1].ud[act_i[2], powered_phase], color=:green)
yvm = Makie.lines!(b7, tact[powered_phase], soln[1].ud[act_i[3], powered_phase], color=:blue)
yvm = Makie.lines!(b7, tact[powered_phase], soln[1].ud[act_i[4], powered_phase], color=:orange)
zvm = Makie.lines!(b7, tact[aero_phase], soln[1].ud[act_i[1], aero_phase], color=:red, linestyle=:dot)
xvm = Makie.lines!(b7, tact[aero_phase], soln[1].ud[act_i[2], aero_phase], color=:green, linestyle=:dot)
yvm = Makie.lines!(b7, tact[aero_phase], soln[1].ud[act_i[3], aero_phase], color=:blue, linestyle=:dot)
yvm = Makie.lines!(b7, tact[aero_phase], soln[1].ud[act_i[4], aero_phase], color=:orange, linestyle=:dot)
b8 = Axis(f[1, 4], title="Acceleration from thrust (m/s^2)")
act_u = [5,6,7]
zvm = Makie.lines!(b8, tact[powered_phase], soln[1].ud[act_u[3], powered_phase], color=:green)
xvm = Makie.lines!(b8, tact[powered_phase], soln[1].ud[act_u[1], powered_phase], color=:blue)
yvm = Makie.lines!(b8, tact[powered_phase], soln[1].ud[act_u[2], powered_phase], color=:red)
zvm = Makie.lines!(b8, tact[aero_phase], soln[1].ud[act_u[3], aero_phase], color=:green, linestyle=:dot)
xvm = Makie.lines!(b8, tact[aero_phase], soln[1].ud[act_u[1], aero_phase], color=:blue, linestyle=:dot)
yvm = Makie.lines!(b8, tact[aero_phase], soln[1].ud[act_u[2], aero_phase], color=:red, linestyle=:dot)
b9 = Axis(f[2, 4], title="Mass (kg)")
zvm = Makie.lines!(b9, tact[powered_phase], soln[1].xd[1, powered_phase] * mdry, color="#56B4E9")
zvm = Makie.lines!(b9, tact[aero_phase], soln[1].xd[1, aero_phase] * mdry, linestyle=:dot, color="#56B4E9")
b10 = Axis(f[3, 3], title="Body angles (°)", xlabel="Time (s)")
rs = map(r->RotXYZ(QuatRotation(r...)), eachrow(soln[1].xd[5:8,:]'))
zvm = Makie.lines!(b10, tact[powered_phase], rad2deg.(map(r -> r.theta1, rs[powered_phase])), color=:green)
xvm = Makie.lines!(b10, tact[powered_phase], rad2deg.(map(r -> r.theta2, rs[powered_phase])), color=:blue)
yvm = Makie.lines!(b10, tact[powered_phase], rad2deg.(map(r -> r.theta3, rs[powered_phase])), color=:red)
zvm = Makie.lines!(b10, tact[aero_phase], rad2deg.(map(r -> r.theta1, rs[aero_phase])), color=:green, linestyle=:dot)
xvm = Makie.lines!(b10, tact[aero_phase], rad2deg.(map(r -> r.theta2, rs[aero_phase])), color=:blue, linestyle=:dot)
yvm = Makie.lines!(b10, tact[aero_phase], rad2deg.(map(r -> r.theta3, rs[aero_phase])), color=:red, linestyle=:dot)
import CairoMakie
save("sim.pdf", f; backend=CairoMakie, size=(1200,900))


Makie.lines(sol[sim.iv], norm.(sol[sim.alpha]))

