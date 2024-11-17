using Rotations
import Quaternions

import CSV
using DataFrames
using DataInterpolations
using Interpolations
using ISAtmosphere
using LinearAlgebra

body_data = CSV.read("src/kerbal/body_fit.csv", DataFrame)
body_data = sort(sort(body_data, order(:mach)), order(:aoa_net))
function make_interp(field)
    field_df = unstack(body_data, :mach, :aoa_net, field)
    aoa_kts = parse.(Float64, names(field_df)[2:end])
    mach_kts = field_df[:,:mach]
    return extrapolate(interpolate((mach_kts, aoa_kts), convert(Matrix{Float64}, Matrix(field_df[:,2:end])), Gridded(Linear())), Flat())
end
body_lift = make_interp(:lift)
body_drag = make_interp(:drag)
body_torque = make_interp(:trq)


actuator_data = CSV.read("src/kerbal/control_eff.csv", DataFrame)
act_lin = extrapolate(interpolate((actuator_data[:, :mach], ), actuator_data[:, :cl1_lin], Gridded(Linear())), Flat())
act_const = extrapolate(interpolate((actuator_data[:, :mach], ), actuator_data[:, :cl1_const], Gridded(Linear())), Flat())
act_scale = extrapolate(interpolate((actuator_data[:, :mach], ), actuator_data[:, :cl1_scale], Gridded(Linear())), Flat())



actuator_lims = CSV.read("src/kerbal/control_lims.csv", DataFrame)
aoa2_grp = groupby(actuator_lims, :aoa2)
function make_lim_lut(field)
    dfs = unstack.(collect(aoa2_grp), :mach, :aoa1, field)
    function decompose_unstacked(df)
        aoa_kts = parse.(Float64, names(df)[2:end])
        mach_kts = df[:,:mach]
        return aoa_kts,mach_kts,convert(Matrix{Float64}, Matrix(df[:,2:end]))
    end
    udfs = decompose_unstacked.(dfs)
    @assert all(ud->ud[1] == udfs[1][1], udfs)
    @assert all(ud->ud[2] == udfs[1][2], udfs)
    aoa1_kts = udfs[1][1]
    aoa2_kts = map(gk->gk.aoa2, keys(aoa2_grp))
    mach_kts = udfs[1][2]
    mat = stack(map(ud->ud[3], udfs)) # goes mach, aoa1, aoa2
    extrapolate(interpolate((mach_kts, aoa1_kts, aoa2_kts), mat, Gridded(Linear())), Flat())
end
upper_lim1_lut = make_lim_lut(:upper_lim1)
lower_lim1_lut = make_lim_lut(:lower_lim1)
upper_lim2_lut = make_lim_lut(:upper_lim2)
lower_lim2_lut = make_lim_lut(:lower_lim2)

struct LimLookupLut{T,TF}
    mach::T
    func::TF
end
struct ErrLookupLut{T1,T2}
    ubl::T1 
    lbl::T2 
end
Base.nameof(::ErrLookupLut) = :ErrLookupLut
function (ll::LimLookupLut)(alpha1, alpha2)
    mach = ll.mach
    return ll.func(mach, alpha1, alpha2)
end

@register_symbolic (e::ErrLookupLut)(u, mach, alpha1, alpha2)
function (e::ErrLookupLut)(u, mach, alpha1, alpha2)
    upper_lim1 = LimLookupLut(mach, e.ubl)(alpha1, alpha2)
    lower_lim1 = LimLookupLut(mach, e.lbl)(alpha1, alpha2)
    return max(-(lower_lim1 + u), 0) + max(u - upper_lim1, 0)
end
SparseConnectivityTracer.is_der1_arg1_zero_global(::LimLookupLut) = false
SparseConnectivityTracer.is_der2_arg1_zero_global(::LimLookupLut) = false
SparseConnectivityTracer.is_der1_arg2_zero_global(::LimLookupLut) = false
SparseConnectivityTracer.is_der2_arg2_zero_global(::LimLookupLut) = false
SparseConnectivityTracer.is_der_cross_zero_global(::LimLookupLut) = false

lim_viol1 = ErrLookupLut(upper_lim1_lut, lower_lim1_lut)
lim_viol2 = ErrLookupLut(upper_lim2_lut, lower_lim2_lut)


#drag1 = (act_lin * cos(alpha2) + act_const) * (act_scale * control)^2
struct AeroLookup1DTs{IT}
    itp::IT 
end
@register_symbolic (a::AeroLookup1DTs)(mach)
(a::AeroLookup1DTs)(mach) = a.itp(mach)
SparseConnectivityTracer.is_der1_zero_global(::AeroLookup1DTs) = false
SparseConnectivityTracer.is_der2_zero_global(::AeroLookup1DTs) = false

act_lin_lookup = AeroLookup1DTs(act_lin)
act_const_lookup = AeroLookup1DTs(act_const)
act_scale_lookup = AeroLookup1DTs(act_scale)

struct AeroLookupTs{IT}
    itp::IT
end
Base.nameof(::AeroLookup1DTs) = :AeroLookup1DTs
Base.nameof(::AeroLookupTs) = :AeroLookupTs
@register_symbolic (a::AeroLookupTs)(mach, alpha)
(a::AeroLookupTs)(mach, alpha) = a.itp(mach, alpha)
body_drag_lookup = AeroLookupTs(body_drag)
body_lift_lookup = AeroLookupTs(body_lift)
body_torq_lookup = AeroLookupTs(body_torque)
SparseConnectivityTracer.is_der1_arg1_zero_global(::AeroLookupTs) = false
SparseConnectivityTracer.is_der2_arg1_zero_global(::AeroLookupTs) = false
SparseConnectivityTracer.is_der1_arg2_zero_global(::AeroLookupTs) = false
SparseConnectivityTracer.is_der2_arg2_zero_global(::AeroLookupTs) = false
SparseConnectivityTracer.is_der_cross_zero_global(::AeroLookupTs) = false


function mkrot(R)
    if length(R) == 4
        return QuatRotation(R[1], R[2], R[3], R[4], false)
    elseif length(R) ==2
        return RotXY(R[1], R[2]) 
    end
end
rquat(R) = mkrot(R)
iquat(R) = inv(mkrot(R))

Rotations.kinematics(r::RotXY, ω) = ω
Base.:/(q::Quaternions.Quaternion, x::Num) = Quaternions.Quaternion(q.s / x, q.v1 / x, q.v2 / x, q.v3 / x)

@register_symbolic mach_vs_temp(T_K)
mach_vs_temp(t_k) = sqrt(ISAtmosphere.κ*ISAtmosphere.R_M²_Ks²*t_k)

@register_symbolic temp_vs_alt(h)
function T_K_fwd(Hp_m::T, ΔT_K::Float64 = 0.0) where T
    ifelse(Hp_m ≤ ISAtmosphere.Hp_trop_m,
        ISAtmosphere.T₀_K + ΔT_K + ISAtmosphere.βT∇_K_m * Hp_m,
        ISAtmosphere.T₀_K + ΔT_K + ISAtmosphere.βT∇_K_m * ISAtmosphere.Hp_trop_m)
end
SparseConnectivityTracer.is_der1_zero_global(::typeof(T_K_fwd)) = false
SparseConnectivityTracer.is_der2_zero_global(::typeof(T_K_fwd)) = false
eval(SparseConnectivityTracer.generate_code_1_to_1(:Main, T_K_fwd))


@register_symbolic p_Pa_fwd(Hp_m::Num)
function p_Pa_fwd(Hp_m, ΔT_K = 0.0)
    ifelse(Hp_m ≤ Hp_trop_m,
        p₀_Pa*((T_K_fwd(Hp_m, ΔT_K) - ΔT_K) / T₀_K) ^ (-g₀_m_s²/(βT∇_K_m*R_M²_Ks²)),
        let p_trop = p₀_Pa*((T_K_fwd(Hp_trop_m, ΔT_K) - ΔT_K) / T₀_K) ^ (-g₀_m_s²/(βT∇_K_m*R_M²_Ks²));
        p_trop * exp(-g₀_m_s² / (R_M²_Ks²*T_K(Hp_trop_m)) * (Hp_m - Hp_trop_m)) end)
end
SparseConnectivityTracer.is_der1_zero_global(::typeof(p_Pa_fwd)) = false
SparseConnectivityTracer.is_der2_zero_global(::typeof(p_Pa_fwd)) = false
eval(SparseConnectivityTracer.generate_code_1_to_1(:Main, p_Pa_fwd))
function ρ_kg_m³_fwd(p_Pa, T_K)
    return p_Pa / (R_M²_Ks² * T_K)
end
@register_symbolic ρ_fun(h)
ρ_fun(h) = ρ_kg_m³_fwd(p_Pa_fwd(h), T₀_K)


temp_vs_alt(h) = T_K_fwd(h)

@register_symbolic sqrt_smooth(a::Num)
function sqrt_smooth(s)
    return ifelse(s == 0.0, s, sqrt(s))
end
SparseConnectivityTracer.is_der1_zero_global(::typeof(sqrt_smooth)) = false
SparseConnectivityTracer.is_der2_zero_global(::typeof(sqrt_smooth)) = false
eval(SparseConnectivityTracer.generate_code_1_to_1(:Main, sqrt_smooth))

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

@register_symbolic logme(v::Real)
logme(v) = let _ = println(v); v end

function normalize_vec(vect)
    return vect/norm(vect)
end


function lookat_rmat(f, p, tgt) # moller and hughes
    u=p.-f
    v=p.-tgt
    (I(3) - 2/dot(u,u) * u * u' - 2/dot(v,v) * v * v' + 4*(dot(u,v))/(dot(u,u)*dot(v,v))*v*u')
end

ksc_lat = -6/60 + 9/3600 # 0d 6m 9s S
angle_from_pole = 90 - ksc_lat # around east/2 axis

function make_vehicle(; 
    iJz_wet=[1/797918.19, 1/797918.19, 1/8197],
    iJz_dry=[1/444490.19, 1/444490.19, 1/3955],
    mdry=10088,
    fuel_mass=10088,
    engine_offset = [0, 0, -8], name)
    @parameters kerbin_center[1:3] = [0.0,0.0,-600000], [tunable = false]
    @parameters kerbin_radius = 600000.0, [tunable = false]
    @parameters iJz_delta[1:3] = iJz_wet - iJz_dry, [tunable = false]#.*u"kg" .* u"m^2")
    @parameters iJz_dry[1:3] = iJz_dry, [tunable = false]#.*u"kg" .* u"m^2")
    @parameters wind[1:3] = [0,0,0], [tunable = false]
    @parameters Cmdot = 4.0, [tunable = false]
    @parameters engine_offset[1:3] = engine_offset, [tunable = false]
    @parameters fin_offset[1:3] = [0, 0, 1], [tunable = false]
    @parameters ISP = 300, [tunable = false] fuel_mass = fuel_mass, [tunable = false] mdry = mdry, [tunable = false]
    @parameters speed_of_sound = 340, [tunable = false] ρ₀=1.225, [tunable = false]
    @parameters ρω[1:2]=ones(3), [tunable = false] ρR[1:2]=ones(2), [tunable = false] ρv[1:3]=ones(3), [tunable = false] ρpos[1:3]=ones(3), [tunable = false]
    @parameters ωk[1:3]=RotY(deg2rad(angle_from_pole)) * [0,0, 2π/21549.425], [tunable=false] # 2 pi radians every 21549s
    @parameters μk=3.5316e12, [tunable=false]

    Symbolics.@variables pos(t)[1:3] v(t)[1:3] m(t) propellant_fraction(t)
    Symbolics.@variables R(t)[1:2] ω(t)[1:2] th(t)[1:3] speed_of_sound(t)

    Symbolics.@variables free_dynamic_pressure(t) phi(t) vel(t) iJz(t)[1:3] alpha1(t) alpha2(t)
    Symbolics.@variables aero_moment(t) aero_torque(t)[1:3] torque(t)[1:3] aero_force(t)[1:3] Cdfs(t) Clfs(t) net_torque(t)[1:3]
    Symbolics.@variables lift_dir(t)[1:3] local_wind_vec(t)[1:3] alpha(t) accel(t)[1:3]
    Symbolics.@variables fin_force(t)[1:4, 1:3] fin1_force(t)[1:3] fin2_force(t)[1:3] fin3_force(t)[1:3] fin4_force(t)[1:3] τc(t) ρᵣ(t)
    Symbolics.@variables u(t)[1:3] kerbin_rel(t)[1:3] spherical_alt(t) temp(t) mach(t) Cd(t) Cl(t) Cm(t) aero_force(t)[1:3] vel_dir(t)[1:3]
    Symbolics.@variables ua(t)[1:2] aero_ctrl_lift(t)[1:2] lift_dir1(t)[1:3] lift_dir2(t)[1:3] aero_ctrl_drag(t) aero_ctrl_force(t)[1:3] body_torque(t)[1:3] ctrl_torque(t)[1:3]
    Symbolics.@variables τc(t) acc(t)[1:3] centrifugal_accel(t)[1:3] coriolis_accel(t)[1:3] g_accel(t)[1:3] earth_equiv_alt(t)
    @parameters τa [tunable = true, dilation=true] τp [tunable=true, dilation=true]
    
    eqns = expand_derivatives.([
        kerbin_rel .~ ρpos.*pos - kerbin_center
        spherical_alt ~ norm(kerbin_rel) - kerbin_radius
        earth_equiv_alt ~ 7963.75*(spherical_alt/1000)/(6371 + 1.25*(spherical_alt/1000))*1000
        temp ~ temp_vs_alt(earth_equiv_alt)
        speed_of_sound ~ mach_vs_temp(temp)
        mach ~ norm(ρv.*v)/speed_of_sound
        alpha ~ angle(v, rquat(ρR .* R) * [0,0,-1]);
        local_wind_vec .~ iquat(ρR .* R) * Symbolics.scalarize(v);
        alpha1 ~ angle_in_plane(vel_dir, rquat(ρR .* R) * [0,0,-1], lift_dir2);
        alpha2 ~ angle_in_plane(vel_dir, rquat(ρR .* R) * [0,0,-1], lift_dir1);
        ρᵣ ~ ρ_kg_m³_fwd(p_Pa_fwd(earth_equiv_alt), T₀_K)/ρ₀

        Cd ~ body_drag_lookup(mach, alpha)
        Cl ~ body_lift_lookup(mach, alpha)
        Cm ~ body_torq_lookup(mach, alpha)

        # todo: velocity correction
        Symbolics.scalarize(aero_ctrl_lift .~ ua .* sum((ρv .* v) .^ 2) .* act_scale_lookup(mach))
        aero_ctrl_drag ~ 
            act_lin_lookup(mach) * sum([cosd(alpha2), cosd(alpha1)] .* aero_ctrl_lift.^2) +
            act_const_lookup(mach) * sum(aero_ctrl_lift.^2)
        Symbolics.scalarize(vel_dir .~ normalize_vec(ρv .* v))
        Symbolics.scalarize(lift_dir1 .~ lookat_rmat([0,0,-1],[0,1,0],Symbolics.scalarize(vel_dir)) * [1,0,0])
        Symbolics.scalarize(lift_dir2 .~ lookat_rmat([0,0,-1],[0,1,0],Symbolics.scalarize(vel_dir)) * [0,1,0])
        #Symbolics.scalarize(lift_dir1 .~ normalize_vec(cross(rquat(ρR .* R) * [0,1,0], ρv .* v)))
        #Symbolics.scalarize(lift_dir2 .~ normalize_vec(cross(ρv .* v, cross(rquat(ρR .* R) * [0,1,0], ρv .* v))))
        Symbolics.scalarize(aero_ctrl_force .~ 1000 * (lift_dir1 .* aero_ctrl_lift[1] .+ lift_dir2 .* aero_ctrl_lift[2] .- aero_ctrl_drag * v/norm(v)))

        aero_force .~ Symbolics.scalarize(
            Cl .* ρᵣ .* 1000 .* (cross(ρv .* v, cross(rquat(ρR .* R) * [0,0,-1],ρv .* v)))
            .- Cd * ρᵣ * 1000 *norm(Symbolics.scalarize(ρv .* v))*(ρv .* v)
            .+ ρᵣ *aero_ctrl_force) # Cl/Cd come out in kN 

        Symbolics.scalarize(body_torque .~ Cm * 1000 * cross([0,0,-1], iquat(ρR .* R) * Symbolics.scalarize(ρv .* v)) * norm(ρv .* v))
        Symbolics.scalarize(ctrl_torque .~ cross([0,0,9.18], iquat(ρR .* R) * Symbolics.scalarize(aero_ctrl_force))) # 9.18 = height up the rocket (in m) of the fins

        iJz .~ iJz_dry # + iJz_delta * propellant_fraction;
        τc ~ 10*ifelse(t >= 0.5, τp, τa)
        th .~ u * 980000
        D(m) ~ -τc*sqrt_smooth(max(sum(th.^2), 0.0))/(ISP * 9.8 * mdry); 
        net_torque .~ cross(engine_offset, th) .+ ρᵣ * (ctrl_torque .+ body_torque); # TODO: change 50
        D.(ρω.*ω) .~ -τc.*collect((Symbolics.scalarize(iquat(ρR.*R) * Symbolics.scalarize(iJz .* net_torque)))[1:2] .+ 10*ω);
        D.(ρR.*R) .~ τc.*Rotations.kinematics(rquat(ρR.*R), Symbolics.scalarize(ρω.*ω));

        centrifugal_accel .~ cross(ωk, cross(ωk, kerbin_rel))
        coriolis_accel .~ 2*cross(ωk, ρv.*v)
        g_accel .~ -kerbin_rel .* μk/Symbolics.scalarize(sum(kerbin_rel.^2)^(3//2))
        acc .~ (g_accel .+ (iquat(ρR .* R) * Symbolics.scalarize(th))/(m*mdry) .+ aero_force/(m*mdry) .- centrifugal_accel .- coriolis_accel)
        Symbolics.scalarize(D.(ρv.*v) .~ τc.*acc);
        Symbolics.scalarize(D.(ρpos.*pos) .~ τc.*(v .* ρv));
    ])
    return ODESystem(expand_derivatives.(Symbolics.scalarize.(eqns)), t; name = name)
end

function build_example_problem()
    @named veh = make_vehicle()
    @named inputx = first_order_hold(N = 20, dt=0.025, tmin=0.5)
    @named inputy = first_order_hold(N = 20, dt=0.025, tmin=0.5)
    @named inputz = first_order_hold(N = 20, dt=0.025, tmin=0.5)

    @named input_fin1 = first_order_hold(N=20, dt=0.05)
    @named input_fin2 = first_order_hold(N=20, dt=0.05)
        
    @named model = ODESystem([
        veh.u[1] ~ inputx.output.u, 
        veh.u[2] ~ inputy.output.u, 
        veh.u[3] ~ inputz.output.u,
        veh.ua[1] ~ input_fin1.output.u,
        veh.ua[2] ~ input_fin2.output.u], t,
        systems = [veh, input_fin1, input_fin2, inputx, inputy, inputz])
    return model # structural_simplify(model)
end

probsys = build_example_problem()
ssys = structural_simplify(probsys)

tf_max = 15.0
tf_min = 0.25
pos_init = [0.0,10000.0,50000.0]
vel_init = [0,-230,-650]
R_init = [-deg2rad(atand(250,575)),0]
ω_init = [0,0]
m_init = (10088 + 10088)/10088

pos_final = [0,0,0.0]
vel_final = [0,0,0.0]
R_final = [0.0,0]
ω_final = [0,0.0]

R_scale = [1.0,1.0]
pos_scale = [10000.0,10000.0,10000.0]
vel_scale = [100.0,100.0,100.0]
ω_scale = [1.0,1.0]
prob = ODEProblem(ssys, [
    ssys.veh.m => m_init
    ssys.veh.ω => ω_init
    ssys.veh.R => R_init ./ R_scale
    ssys.veh.pos => pos_init ./ pos_scale
    ssys.veh.v => vel_init ./ vel_scale
    ssys.veh.τa => 120.0/10
    ssys.veh.τp => 20.0/10
], (0.0, 1.0), [
    ssys.veh.ρv => vel_scale
    ssys.veh.ρpos => pos_scale
    ssys.input_fin1.vals => 0.0*ones(20)
    ssys.input_fin2.vals => -0.0*ones(20) # 0.0*ones(20) #
    ssys.inputz.vals => 0.1 * ones(20)
])
sol = solve(prob, Tsit5(); dtmax=0.0001)

function mkinit_interp(var, N, sol)
    sol(LinRange(0.0,1.0,N), idxs=var).u
end
mkinit(var, N, sol) = eachrow(stack(mkinit_interp(var, N, sol)))
lin_range_vals(i,f,n) = eachrow(reduce(hcat, LinRange(i, f, n)))
prb = trajopt(probsys, (0.0, 1.0), 41, 
    Dict([
        probsys.veh.ρv => vel_scale,
        probsys.veh.ρpos => pos_scale,
        probsys.veh.ρR => R_scale,
        probsys.veh.ρω => ω_scale,
        probsys.veh.τa => 75.0/10,
        probsys.veh.τp => 15.0/10
    ]), 
    Dict(
        [probsys.veh.m => collect(LinRange(m_init, m_init, 41));
        Symbolics.scalarize(probsys.veh.pos .=> mkinit(probsys.veh.pos, 41, sol)); #lin_range_vals(pos_init ./ pos_scale, pos_final, 41));
        Symbolics.scalarize(probsys.veh.v .=> mkinit(probsys.veh.v, 41, sol)); # lin_range_vals(vel_init ./ vel_scale, vel_final ./ vel_scale, 41));
        Symbolics.scalarize(probsys.veh.R .=> mkinit(probsys.veh.R, 41, sol)); # lin_range_vals(R_init ./ R_scale, R_final ./ R_scale, 41));
        Symbolics.scalarize(probsys.veh.ω .=> mkinit(probsys.veh.ω, 41, sol)); # lin_range_vals(ω_init ./ ω_scale, ω_final ./ ω_scale, 41))
        ]), 
    [probsys.veh.m => m_init,
    probsys.veh.pos => pos_init ./ pos_scale,
    probsys.veh.v => vel_init ./ vel_scale,
    probsys.veh.R => R_init ./ R_scale,
    probsys.veh.ω => ω_init ./ ω_scale
    ], 
    (probsys.veh.τa + probsys.veh.τp)/5 + sqrt_smooth(Symbolics.scalarize(sum(probsys.veh.u .^ 2))), 0,# -100*dot(probsys.veh.pos .* pos_scale, [1.0,0.0,0.0]), 
    max(tanh(Symbolics.scalarize(norm(probsys.veh.v))) * (probsys.veh.alpha - 25.0), 0.0) +
    #norm(probsys.veh.ρv .* probsys.veh.v) * probsys.veh.alpha + 
    10*ifelse(t>0.5, max(0.5^2 - Symbolics.scalarize(sum(probsys.veh.u .^ 2)), 0.0), 0.0) + 
    10*lim_viol1(probsys.veh.ua[1], probsys.veh.mach, probsys.veh.alpha1, probsys.veh.alpha2) + 
    10*lim_viol2(probsys.veh.ua[2], probsys.veh.mach, probsys.veh.alpha1, probsys.veh.alpha2), 0.0, # todo: alpha_max_aero (probsys.veh.alpha - 25.0)/50 - need to do expanded dynamics for the pdg phase
    sum((probsys.veh.pos .* pos_scale/100).^2) + ((sum((vel_scale[1:3] .* probsys.veh.v[1:3]).^2))) + sum((probsys.veh.ω) .^2) + sum((probsys.veh.R .* R_scale .- R_final) .^2),
    (tsys) -> begin 
        get_x = getp(tsys, tsys.model.inputx.vals)
        get_y = getp(tsys, tsys.model.inputy.vals)
        get_z = getp(tsys, tsys.model.inputz.vals)
        return function (model, δx, xref, symbolic_params, objexp)
            x_ctrl = get_x(symbolic_params)
            y_ctrl = get_y(symbolic_params)
            z_ctrl = get_z(symbolic_params)
            for (x,y,z) in Iterators.zip(x_ctrl, y_ctrl, z_ctrl)
                # 10.5 degree tvc
                @constraint(model, [z * tand(10.5/2), x, y] in SecondOrderCone())
                # max throttle
                @constraint(model, [1.0, x, y, z] in SecondOrderCone())
            end
            return objexp
        end
    end);

    
    @profview 
    ui,xi,_,_,_,_,_,unk,_ = do_trajopt(prb; maxsteps=1);
    u,x,wh,ch,rch,dlh,lnz,unk,tp = do_trajopt(prb; maxsteps=40);
    @profview u,x,wh,ch,rch,dlh,lnz,unk,tp = do_trajopt(prb; maxsteps=40);
    @profview u,x,wh,ch,rch,dlh,lnz,unk,tp = do_trajopt(prb; maxsteps=300);
    ignst = x[end][:,21]
    ignpt = ignst[11:13]
    pushdir = [1.0, 0.0, 0.0]


    prob_ws = ODEProblem(ssys, [
        ssys.veh.m => m_init
        ssys.veh.ω => ω_init
        ssys.veh.R => R_init
        ssys.veh.pos => pos_init ./ pos_scale
        ssys.veh.v => vel_init ./ vel_scale
    ], (0.0, 1.0), [
        ssys.veh.ρv => vel_scale;
        ssys.veh.ρpos => pos_scale;
        denamespace.((ssys, ), tp) .=> [
            u[end][1], #10*u[end][1], 
            u[end][2], 
            u[end][3:22], 
            u[end][23:42], 
            u[end][43:62], 
            u[end][63:82], 
            u[end][83:102]]
    ])
    sol_ws = solve(prob_ws, Tsit5(); dtmax=0.001)

prb_divert = trajopt(probsys, (0.0, 1.0), 41, 
    Dict([
        probsys.veh.ρv => vel_scale,
        probsys.veh.ρpos => pos_scale,
        probsys.veh.ρR => R_scale,
        probsys.veh.ρω => ω_scale,
        probsys.veh.τa => 75.0/10,
        probsys.veh.τp => 15.0/10
    ]), 
    Dict(
        [probsys.veh.m => collect(LinRange(m_init, m_init, 41));
        Symbolics.scalarize(probsys.veh.pos .=> mkinit(probsys.veh.pos, 41, sol)); #lin_range_vals(pos_init ./ pos_scale, pos_final, 41));
        Symbolics.scalarize(probsys.veh.v .=> mkinit(probsys.veh.v, 41, sol)); # lin_range_vals(vel_init ./ vel_scale, vel_final ./ vel_scale, 41));
        Symbolics.scalarize(probsys.veh.R .=> mkinit(probsys.veh.R, 41, sol)); # lin_range_vals(R_init ./ R_scale, R_final ./ R_scale, 41));
        Symbolics.scalarize(probsys.veh.ω .=> mkinit(probsys.veh.ω, 41, sol)); # lin_range_vals(ω_init ./ ω_scale, ω_final ./ ω_scale, 41))
        ]), 
    [probsys.veh.m => m_init,
    probsys.veh.pos => pos_init ./ pos_scale,
    probsys.veh.v => vel_init ./ vel_scale,
    probsys.veh.R => R_init ./ R_scale,
    probsys.veh.ω => ω_init ./ ω_scale
    ], 
    (probsys.veh.τa + probsys.veh.τp)/5 + sqrt_smooth(Symbolics.scalarize(sum(probsys.veh.u .^ 2))), 0,# -100*dot(probsys.veh.pos .* pos_scale, [1.0,0.0,0.0]), 
    max(tanh(Symbolics.scalarize(norm(probsys.veh.v))) * (probsys.veh.alpha - 25.0), 0.0) +
    #norm(probsys.veh.ρv .* probsys.veh.v) * probsys.veh.alpha + 
    10*ifelse(t>0.5, max(0.5^2 - Symbolics.scalarize(sum(probsys.veh.u .^ 2)), 0.0), 0.0), 0.0, # todo: alpha_max_aero (probsys.veh.alpha - 25.0)/50 - need to do expanded dynamics for the pdg phase
    sum((probsys.veh.pos .* pos_scale/100).^2) + ((sum((vel_scale[1:3] .* probsys.veh.v[1:3]).^2))) + sum((probsys.veh.ω) .^2) + sum((probsys.veh.R .* R_scale .- R_final) .^2),
    (tsys) -> begin 
        get_x = getp(tsys, tsys.model.inputx.vals)
        get_y = getp(tsys, tsys.model.inputy.vals)
        get_z = getp(tsys, tsys.model.inputz.vals)

        get_pos = getu(tsys, tsys.model.veh.pos)
        get_push_dir = getp(tsys, tsys.push_dir)
        get_ignpt = getp(tsys, tsys.ignpt)
        return function (model, δx, xref, symbolic_params, objexp)
            x_ctrl = get_x(symbolic_params)
            y_ctrl = get_y(symbolic_params)
            z_ctrl = get_z(symbolic_params)

            # only the tunables in symbolic params are actually symbolic; the rest are just numbers
            push_dir = get_push_dir(symbolic_params)
            ignpt = get_ignpt(symbolic_params)

            JuMP.@variable(model, c)
            @constraint(model, get_pos(δx[:, 21] .+ xref[:, 21]) .== ignpt .+ c .*push_dir)
            for (x,y,z) in Iterators.zip(x_ctrl, y_ctrl, z_ctrl)
                # 10.5 degree tvc
                @constraint(model, [z * tand(10.5/2), x, y] in SecondOrderCone())
                # max throttle
                @constraint(model, [1.0, x, y, z] in SecondOrderCone())
            end
            return objexp - 10*c
        end
    end,
    (sys, l, y) -> begin 
        @parameters push_dir[1:3]=[1,0,0], [tunable=false,input=true]
        @parameters ignpt[1:3]=[0,0,0], [tunable=false,input=true]
        function add_divert!(i, u, p, c)
            i[u.l] -= 10*dot([i.u[u.posx], i.u[u.posy], i.u[u.posz]] .- i.ps[p.ignpt], i.ps[p.push_dir])
        end
        @named augmenting=ODESystem(Equation[], t, [], [push_dir,ignpt]; discrete_events=[
            [0.5] => (add_divert!, [l => :l, probsys.veh.pos[1] => :posx, probsys.veh.pos[2] => :posy, probsys.veh.pos[3] => :posz], [push_dir,ignpt], [], nothing)
        ])
        return extend(sys, augmenting)
    end);
    ui,xi,_,_,_,_,_,unk,_ = do_trajopt(prb_divert; maxsteps=1);

    upd_push_dir = setp(prb_divert[:tsys], prb_divert[:tsys].push_dir)
    upd_ignpt = setp(prb_divert[:tsys], prb_divert[:tsys].ignpt)

    dir_rand = rand(3)
    dir_rand = dir_rand/norm(dir_rand)
    upd_push_dir(prb_divert[:pars], [0,0,1])
    upd_ignpt(prb_divert[:pars], ignpt)
    up,xp,whp,chp,rchp,dlhp,lnzp,unkp,tpp = do_trajopt(prb_divert; maxsteps=50);

    pushed = []

    for i=1:10
        dir_rand = rand(3)
        dir_rand = dir_rand/norm(dir_rand)
        upd_push_dir(prb_divert[:pars], dir_rand)
        up,xp,whp,chp,rchp,dlhp,lnzp,unkp,tpp = do_trajopt(prb_divert; maxsteps=25);
        push!(pushed, xp[end][11:13,21])
    end

    
f = Figure()
ax = Makie.Axis(f[1,1])
for i in eachindex(x[1:5:end])
    lines!(ax, x[i][end-10,:])
end
f


prob = ODEProblem(ssys, [
    ssys.veh.m => m_init
    ssys.veh.ω => ω_init
    ssys.veh.R => R_init
    ssys.veh.pos => pos_init ./ pos_scale
    ssys.veh.v => vel_init ./ vel_scale
], (0.0, 1/20.0), [
    ssys.veh.ρv => vel_scale;
    ssys.veh.ρpos => pos_scale;
    denamespace.((ssys, ), tp) .=> [
        u[end][1], #10*u[end][1], 
        u[end][2:21], 
        u[end][22:end]]
])
sol = solve(prob, Tsit5(); dtmax=0.001)


prob_res = ODEProblem(ssys, [
    ssys.veh.m => m_init
    ssys.veh.ω => ω_init
    ssys.veh.R => R_init
    ssys.veh.pos => pos_init ./ pos_scale
    ssys.veh.v => vel_init ./ vel_scale
], (0.0, 1.0), [
    ssys.veh.ρv => vel_scale;
    ssys.veh.ρpos => pos_scale;
    denamespace.((ssys, ), tp) .=> [
        u[end][1], #10*u[end][1], 
        u[end][2], 
        u[end][3:22], 
        u[end][23:42], 
        u[end][43:62], 
        u[end][63:82], 
        u[end][83:102]]
])
prob_res = ODEProblem(ssys, [
    ssys.veh.m => m_init
    ssys.veh.ω => ω_init
    ssys.veh.R => R_init
    ssys.veh.pos => pos_init ./ pos_scale
    ssys.veh.v => vel_init ./ vel_scale
], (0.0, 1.0), [
    ssys.veh.ρv => vel_scale;
    ssys.veh.ρpos => pos_scale;
    denamespace.((ssys, ), tp) .=> [
        up[end][1], #10*up[end][1], 
        up[end][2], 
        up[end][3:22], 
        up[end][23:42], 
        up[end][43:62], 
        up[end][63:82], 
        up[end][83:102]]
])

sol_res = solve(prob_res, Tsit5())

retimer(t) = 10*(min(t, 0.5) * prob.ps[ssys.veh.τa] + max(t - 0.5, 0) * prob.ps[ssys.veh.τp])

f=Figure()
ax1=Makie.Axis(f[1,1])
timebase = retimer.(sol_res.t)
lines!(ax1, timebase, (sol_res[ssys.veh.ua[1]]))
lines!(ax1, timebase, (sol_res[ssys.veh.ua[2]]))
ax2=Makie.Axis(f[2,1])
lines!(ax2, timebase, (sol_res[ssys.veh.alpha1]))
lines!(ax2, timebase, (sol_res[ssys.veh.alpha2]))
ax3=Makie.Axis(f[3,1])
lines!(ax3, timebase, (sol_res[ssys.veh.pos[1]]))
lines!(ax3, timebase, (sol_res[ssys.veh.pos[2]]))
ax4=Makie.Axis(f[4,1])
lines!(ax4, timebase, (sol_res[ssys.veh.aero_force[1]]))
lines!(ax4, timebase, (sol_res[ssys.veh.aero_force[2]]))
lines!(ax4, timebase, (sol_res[ssys.veh.aero_force[3]]))
f
lines(timebase, (sol[Symbolics.scalarize(norm(ssys.veh.aero_force/(ssys.veh.m *9.8*ssys.veh.m*ssys.veh.mdry)))]))
lines(timebase, (sol_res[ssys.veh.u[3]]))
lines!(timebase, (sol_res[ssys.veh.u[1]]))
lines!(timebase, (sol_res[ssys.veh.u[2]]))
lines(timebase, (sol_res[Symbolics.scalarize(norm(ssys.veh.u))]))

lines(Point3.(sol[ssys.veh.pos]))
scatter!(Point3.(pushed))


pos_init_2 = [0, 40000, 30000]
vel_init_2 = [0, -1200, -750]
rot_axis = cross(vel_init_2, [0,0,-1])/norm(vel_init_2)
rot = QuatRotation(AngleAxis(-asin(norm(rot_axis)), rot_axis..., true))
R_init_2 = [rot.q.s, rot.q.v1, rot.q.v2, rot.q.v3]
prob2 = ODEProblem(ssys, [
    ssys.veh.m => m_init
    ssys.veh.ω => ω_init
    ssys.veh.R => R_init_2
    ssys.veh.pos => pos_init_2 ./ pos_scale
    ssys.veh.v => vel_init_2 ./ vel_scale
], (0.0, 1.0), [
    ssys.veh.ρv => vel_scale;
    ssys.veh.ρpos => pos_scale;
    denamespace.((ssys, ), tp) .=> [
        45, #10*u[end][1], 
        0.0 * ones(20), 
        0.0*ones(20)]
])
sol = solve(prob2, Tsit5(); dtmax=0.001)

f=Figure()
ax = Makie.Axis(f[1,1])
lines!(ax, sol.t, sol[ssys.veh.pos[2]])
lines!(ax, sol.t, sol[ssys.veh.pos[3]])
f
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

