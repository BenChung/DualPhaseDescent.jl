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

ksc_lat = -(6/60 + 9/3600) # 0d 6m 9s S
angle_from_pole = 90 - ksc_lat # around east/2 axis

function make_vehicle(; 
    iJz_wet=[1/555803, 1/555803, 1/9105],
    iJz_dry=[1/466296, 1/466296, 1/5326],
    mdry=10516.0,
    fuel_mass=19516.0 - 10516.0,
    engine_offset = [0, 0, -4], name)
    @parameters kerbin_center[1:3] = [0.0,0.0,-600000], [tunable = false]
    @parameters kerbin_radius = 600000.0, [tunable = false]
    @parameters iJz_delta[1:3] = iJz_wet - iJz_dry, [tunable = false]#.*u"kg" .* u"m^2")
    @parameters iJz_dry[1:3] = iJz_dry, [tunable = false]#.*u"kg" .* u"m^2")
    @parameters wind[1:3] = [0,0,0], [tunable = false]
    @parameters Cmdot = 4.0, [tunable = false]
    @parameters engine_offset[1:3] = engine_offset, [tunable = false]
    @parameters fin_offset[1:3] = [0, 0, 9.2], [tunable = false]
    @parameters ISP = 300, [tunable = false] fuel_mass = fuel_mass, [tunable = false] mdry = mdry, [tunable = false]
    @parameters ρ₀=1.225, [tunable = false]

    @parameters ρω[1:2]=ones(3), [tunable = false] ρR[1:2]=ones(2), [tunable = false] ρv[1:3]=ones(3), [tunable = false] ρpos[1:3]=ones(3), [tunable = false] ρm=mdry, [tunable = false]

    @parameters ωk[1:3]=RotY(deg2rad(angle_from_pole)) * [0,0, 2π/21549.425], [tunable=false] # 2 pi radians every 21549s
    @parameters μk=3.5316e12, [tunable=false]

    Symbolics.@variables pos(t)[1:3] v(t)[1:3] m(t) propellant_fraction(t)
    Symbolics.@variables R(t)[1:2] ω(t)[1:2] th(t)[1:3] speed_of_sound(t)

    Symbolics.@variables ωp(t)[1:2] Rp(t)[1:2] vp(t)[1:3] posp(t)[1:3] mp(t)

    Symbolics.@variables free_dynamic_pressure(t) phi(t) vel(t) iJz(t)[1:3] alpha1(t) alpha2(t)
    Symbolics.@variables aero_moment(t) aero_torque(t)[1:3] torque(t)[1:3] aero_force(t)[1:3] Cdfs(t) Clfs(t) net_torque(t)[1:3]
    Symbolics.@variables lift_dir(t)[1:3] local_wind_vec(t)[1:3] alpha(t) accel(t)[1:3]
    Symbolics.@variables fin_force(t)[1:4, 1:3] fin1_force(t)[1:3] fin2_force(t)[1:3] fin3_force(t)[1:3] fin4_force(t)[1:3] τc(t) ρᵣ(t)
    Symbolics.@variables u(t)[1:3] kerbin_rel(t)[1:3] spherical_alt(t) temp(t) mach(t) Cd(t) Cl(t) Cm(t) aero_force(t)[1:3] vel_dir(t)[1:3]
    Symbolics.@variables ua(t)[1:2] aero_ctrl_lift(t)[1:2] lift_dir1(t)[1:3] lift_dir2(t)[1:3] aero_ctrl_drag(t) aero_ctrl_force(t)[1:3] body_torque(t)[1:3] ctrl_torque(t)[1:3]
    Symbolics.@variables τc(t) acc(t)[1:3] centrifugal_accel(t)[1:3] coriolis_accel(t)[1:3] g_accel(t)[1:3] earth_equiv_alt(t)
    @parameters τa [tunable = true, dilation=true] τp [tunable=true, dilation=true]
    
    eqns = expand_derivatives.([
        ωp .~ ρω .* ω
        Rp .~ ρR .* R
        vp .~ ρv .* v
        posp .~ ρpos .* pos
        mp ~ ρm * m

        kerbin_rel .~ posp - kerbin_center
        spherical_alt ~ norm(kerbin_rel) - kerbin_radius
        earth_equiv_alt ~ 7963.75*(spherical_alt/1000)/(6371 + 1.25*(spherical_alt/1000))*1000
        temp ~ temp_vs_alt(earth_equiv_alt)
        speed_of_sound ~ mach_vs_temp(temp)
        mach ~ norm(vp)/speed_of_sound
        alpha ~ angle(v, rquat(Rp) * [0,0,-1]);
        local_wind_vec .~ iquat(Rp) * Symbolics.scalarize(v);
        alpha1 ~ angle_in_plane(vel_dir, rquat(Rp) * [0,0,-1], lift_dir2);
        alpha2 ~ angle_in_plane(vel_dir, rquat(Rp) * [0,0,-1], lift_dir1);
        ρᵣ ~ ρ_kg_m³_fwd(p_Pa_fwd(earth_equiv_alt), T₀_K)/ρ₀

        Cd ~ body_drag_lookup(mach, alpha)
        Cl ~ body_lift_lookup(mach, alpha)
        Cm ~ body_torq_lookup(mach, alpha)

        # todo: velocity correction
        Symbolics.scalarize(aero_ctrl_lift .~ ua .* sum(vp .^ 2) .* act_scale_lookup(mach))
        aero_ctrl_drag ~ 
            act_lin_lookup(mach) * sum([cosd(alpha2), cosd(alpha1)] .* aero_ctrl_lift.^2) +
            act_const_lookup(mach) * sum(aero_ctrl_lift.^2)
        Symbolics.scalarize(vel_dir .~ normalize_vec(vp))
        Symbolics.scalarize(lift_dir1 .~ lookat_rmat([0,0,-1],[0,1,0],Symbolics.scalarize(vel_dir)) * [1,0,0])
        Symbolics.scalarize(lift_dir2 .~ lookat_rmat([0,0,-1],[0,1,0],Symbolics.scalarize(vel_dir)) * [0,1,0])
        #Symbolics.scalarize(lift_dir1 .~ normalize_vec(cross(rquat(Rp) * [0,1,0], vp)))
        #Symbolics.scalarize(lift_dir2 .~ normalize_vec(cross(vp, cross(rquat(Rp) * [0,1,0], vp))))
        Symbolics.scalarize(aero_ctrl_force .~ 1000 * (lift_dir1 .* aero_ctrl_lift[1] .+ lift_dir2 .* aero_ctrl_lift[2] .- aero_ctrl_drag * v/norm(v)))
        
        aero_force .~ Symbolics.scalarize(
            Cl .* ρᵣ .* 1000 .* (cross(vp, cross(rquat(Rp) * [0,0,-1],vp)))
            .- Cd * ρᵣ * 1000 *norm(Symbolics.scalarize(vp))*(vp)
            .+ ρᵣ *aero_ctrl_force) # Cl/Cd come out in kN 

        Symbolics.scalarize(body_torque .~ Cm * 1000 * cross([0,0,-1], iquat(Rp) * Symbolics.scalarize(vp)) * norm(vp))
        Symbolics.scalarize(ctrl_torque .~ cross([0,0,9.18], iquat(Rp) * Symbolics.scalarize(aero_ctrl_force))) # 9.18 = height up the rocket (in m) of the fins

        propellant_fraction ~ (mp - mdry)/fuel_mass
        iJz .~ iJz_dry + iJz_delta * propellant_fraction;
        τc ~ 10*ifelse(t >= 0.5, τp, τa)
        th .~ u * 980000
        D(ρm * m) ~ -τc*sqrt_smooth(max(sum(th.^2), 0.0))/(ISP * 9.8); 
        net_torque .~ cross(engine_offset, th) .+ ρᵣ * (ctrl_torque .+ body_torque); # TODO: change 50
        D.(ρω.*ω) .~ -τc.*collect((Symbolics.scalarize(iquat(Rp) * Symbolics.scalarize(iJz .* net_torque)))[1:2] .+ 10*ωp);
        D.(ρR.*R) .~ τc.*Rotations.kinematics(rquat(Rp), Symbolics.scalarize(ωp));

        centrifugal_accel .~ cross(ωk, cross(ωk, kerbin_rel))
        coriolis_accel .~ 2*cross(ωk, vp)
        g_accel .~ -kerbin_rel .* μk/Symbolics.scalarize(sum(kerbin_rel.^2)^(3//2))
        acc .~ (g_accel .+ (iquat(Rp) * Symbolics.scalarize(th))/(mp) .+ aero_force/(mp) .- centrifugal_accel .- coriolis_accel)
        Symbolics.scalarize(D.(ρv.*v) .~ τc.*acc);
        Symbolics.scalarize(D.(ρpos.*pos) .~ τc.*(vp));
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