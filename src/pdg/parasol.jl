#=
aero_ctrl = ctrl_torque + body_torque
cross(iquat(Rp) * Symbolics.scalarize(aero_ctrl_force), fin_offset) = aero_ctrl - body_torque

a = fin_offset
c = aero_ctrl - body_torque = (in the case of ctrl input) ctrl_torque
iquat(Rp) * aero_ctrl_force = -cross(c, a)/dot(a,a) - t * a
aero_ctrl_force = quat(Rp)*(-cross(c, a)/dot(a,a)) - t * quat(Rp)*(a)

define
lift_scale = sum(vp .^ 2) .* act_scale_lookup(mach)
aero_ctrl_lift .~ ua .* lift_scale
aero_ctrl_drag = act_lin_lookup(mach) * (cosd(alpha2) * aero_ctrl_lift[1]^2 + cosd(alpha1) * aero_ctrl_lift[2]^2) +
            act_const_lookup(mach) * (aero_ctrl_lift[1]^2 + aero_ctrl_lift[2]^2)
= (act_lin_lookup(mach) * cosd(alpha2) + act_const_lookup(mach)) * aero_ctrl_lift[1]^2 + 
  (act_lin_lookup(mach) * cosd(alpha1) + act_const_lookup(mach)) * aero_ctrl_lift[2]^2
= (act_lin_lookup(mach) * cosd(alpha2) + act_const_lookup(mach)) * lift_scale^2 * ua[1]^2 + 
  (act_lin_lookup(mach) * cosd(alpha1) + act_const_lookup(mach)) * lift_scale^2 * ua[2]^2
define
    coupling_1 = (act_lin_lookup(mach) * cosd(alpha2) + act_const_lookup(mach)) * lift_scale
    coupling_2 = (act_lin_lookup(mach) * cosd(alpha1) + act_const_lookup(mach)) * lift_scale
so
aero_ctrl_drag = coupling_1 * lift_scale* ua[1]^2 + coupling_2 * lift_scale* ua[2]^2

aero_ctrl_force = 1000 * (lift_dir1 .* aero_ctrl_lift[1] .+ lift_dir2 .* aero_ctrl_lift[2] .- aero_ctrl_drag * v/norm(v))
= 1000*lift_dir1*lift_scale*ua[1] + 1000*lift_dir2*lift_scale*ua[2] - 1000*(coupling_1 * lift_scale * ua[1]^2 + coupling_2 * lift_scale* ua[2]^2)* v/norm(v)


normalized_ctrl_force/(1000 * lift_scale) = lift_dir1*ua[1] + lift_dir2*ua[2] - (coupling_1 * ua[1]^2 + coupling_2 * ua[2]^2)* v/norm(v)
written in lift_dir1/lift_dir2/v frame, solutions to
    normalized_ctrl_force/(1000 * lift_scale) = [x, y, coupling_1 * x^2 + coupling_2 * y^2]

rotation matrix from body frame to wind frame: Rbw
 -1/(dot(a,a) * 1000 * lift_scale) * Rbw * cross(c, a) - t * Rbw * a/(1000 * lift_scale) = [x, y, coupling_1 * x^2 + coupling_2 * y^2] 


ray-parabolid intersection
  P = C + t * D
  z + k = k1 * x^2 + k2 * y^2
  C[3] + t*D[3] + k = (C[1]^2)*k1 + (C[2]^2)*k2 + (2C[1]*D[1]*k1 + 2C[2]*D[2]*k2)*t + ((D[1]^2)*k1 + (D[2]^2)*k2)*(t^2)
  0 = (C[1]^2)*k1 + (C[2]^2)*k2 - C[3] - k + (2C[1]*D[1]*k1 + 2C[2]*D[2]*k2 - D[3])*t + ((D[1]^2)*k1 + (D[2]^2)*k2)*(t^2)
  
  in the form a t^2 + b t + c = 0
  a = (D[1]^2)*k1 + (D[2]^2)*k2
  b = 2C[1]*D[1]*k1 + 2C[2]*D[2]*k2 - D[3]
  c = (C[1]^2)*k1 + (C[2]^2)*k2 - C[3] - k

  solutions: (-b +- sqrt(b^2-4*a*c))/(2a)

  if a is very small,
       b t + c = 0
  =>   t = -c/b

  C = -1/(dot(a,a) * 1000 * lift_scale) * Rbw * cross(c, a)
  D = -Rbw * a/(1000 * lift_scale)
  k1 = coupling_1
  k2 = coupling_2
  k = 0

=#
@register_symbolic parasol(C::Vector, D::Vector, k, k1, k2)
function parasol(C, D, k, k1, k2)
    a = (D[1]^2)*k1 + (D[2]^2)*k2
    b = 2*C[1]*D[1]*k1 + 2*C[2]*D[2]*k2 - D[3]
    c = (C[1]^2)*k1 + (C[2]^2)*k2 - C[3] - k
    if abs(a) < 1e-20
        return -c/b
    elseif 4*a*c > b^2
        return -b/(2*a)
    else
        return (-b - sqrt(b^2 - 4*a*c))/(2*a)
    end
end