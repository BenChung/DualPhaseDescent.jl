

function mkinit_interp(var, N, sol)
    sol(LinRange(0.0,1.0,N), idxs=var).u
end
mkinit(var, N, sol) = eachrow(stack(mkinit_interp(var, N, sol)))
lin_range_vals(i,f,n) = eachrow(reduce(hcat, LinRange(i, f, n)))
function descentproblem(probsys, sol, solsys; cvx_mod=s->(_, _, _, _, p)->(p, () -> 0.0, (_) -> 0.0, (_) -> 0.0), custsys=(s,l,y)->s)
    vel_scale = sol.ps[solsys.veh.ρv]
    pos_scale = sol.ps[solsys.veh.ρpos]
    R_scale = sol.ps[solsys.veh.ρR]
    @parameters thmin=0.5, [tunable=false, description="Normalized min throttle limit"]
    @parameters sfins=10.0, [tunable=false, description="Fin constraint violation penalty scale"]
    @parameters sth=20.0, [tunable=false, description="Thrust lower bound violation penalty scale"]
    @parameters gimbal_angle=10.5, [tunable=false, description="Gimbal angle off of centerline; degrees"]
    @parameters obj_weight_fuel=1.0, [tunable=false, description="Objective weight on fuel"]
    @parameters obj_weight_time=0.0, [tunable=false, description="Objective weight on time"]
    @parameters obj_weight_ω=0.0, [tunable=false, description="Objective weight on angular rate"]
    @parameters sωmax=0, [tunable=false, description="Maximum pitch rate constraint violation scale"]
    @parameters ωmax=20.0, [tunable=false, description="Maximum pitch rate (deg/s)"]
    @parameters sqmax=0, [tunable=false, description="Maximum aerodynamic pressure constraint violation scale"]
    @parameters qmax=100000, [tunable=false, description="Maximum dynamic pressure Pa"]
    @parameters sqαmax=0, [tunable=false, description="Maximum aerodynamic pressure constraint violation scale"]
    @parameters qαmax=1e6, [tunable=false, description="Maximum aoa dynamic pressure product deg Pa"]
    @named paramsys = ODESystem(Equation[], t, [], [thmin, sfins, sth, gimbal_angle, obj_weight_fuel, obj_weight_time])
    return trajopt(extend(probsys, paramsys), (0.0, 1.0), 41, 
        Dict([
            probsys.veh.ρv => sol.ps[solsys.veh.ρv],
            probsys.veh.ρpos => sol.ps[solsys.veh.ρpos],
            probsys.veh.ρR => sol.ps[solsys.veh.ρR],
            probsys.veh.ρω => sol.ps[solsys.veh.ρω],
            probsys.veh.τa => getp(solsys, solsys.veh.τa)(sol),
            probsys.veh.τp => getp(solsys, solsys.veh.τp)(sol)
        ]), 
        Dict(
            [probsys.veh.m => first.(mkinit(probsys.veh.m, 41, sol));
            Symbolics.scalarize(probsys.veh.pos .=> mkinit(solsys.veh.pos, 41, sol)); #lin_range_vals(pos_init ./ pos_scale, pos_final, 41));
            Symbolics.scalarize(probsys.veh.v .=> mkinit(solsys.veh.v, 41, sol)); # lin_range_vals(vel_init ./ vel_scale, vel_final ./ vel_scale, 41));
            Symbolics.scalarize(probsys.veh.R .=> mkinit(solsys.veh.R, 41, sol)); # lin_range_vals(R_init ./ R_scale, R_final ./ R_scale, 41));
            Symbolics.scalarize(probsys.veh.ω .=> mkinit(solsys.veh.ω, 41, sol)); # lin_range_vals(ω_init ./ ω_scale, ω_final ./ ω_scale, 41))
            ]), 
        [probsys.veh.m => sol(0.0, idxs=solsys.veh.m),
        probsys.veh.pos => sol(0.0, idxs=solsys.veh.pos),
        probsys.veh.v => sol(0.0, idxs=solsys.veh.v),
        probsys.veh.R => sol(0.0, idxs=solsys.veh.R),
        probsys.veh.ω => sol(0.0, idxs=solsys.veh.ω)
        ], 
        obj_weight_time*probsys.veh.τc + obj_weight_fuel*sqrt_smooth(Symbolics.scalarize(sum(probsys.veh.u .^ 2))) + obj_weight_ω * sqrt_smooth(Symbolics.scalarize(sum(probsys.veh.ω .^ 2))), 0,# -100*dot(probsys.veh.pos .* pos_scale, [1.0,0.0,0.0]), 
        [
            max(probsys.veh.alpha - Symbolics.scalarize(25.0/tanh(norm(probsys.veh.v) + 1e-5)), 0.0),
            #max(probsys.veh.alpha - Symbolics.scalarize(qαmax/(probsys.veh.q + 1e-5)), 0.0) +
            #norm(probsys.veh.ρv .* probsys.veh.v) * probsys.veh.alpha + 
            sth*ifelse(t>0.5, max(thmin^2 - Symbolics.scalarize(sum(probsys.veh.u .^ 2)), 0.0), 0.0),
            sfins*lim_viol1(probsys.veh.ua[1], probsys.veh.mach, probsys.veh.alpha1, probsys.veh.alpha2),
            sfins*lim_viol2(probsys.veh.ua[2], probsys.veh.mach, probsys.veh.alpha1, probsys.veh.alpha2),
            sωmax*max(deg2rad(ωmax)^2 - Symbolics.scalarize(sum(probsys.veh.ω .^2)), 0),
            max(sqmax*probsys.veh.q - sqmax*qmax, 0), 
            sqαmax*max(probsys.veh.q * probsys.veh.alpha - qαmax, 0)
        ], #0.0, # todo: alpha_max_aero (probsys.veh.alpha - 25.0)/50 - need to do expanded dynamics for the pdg phase
        Symbolics.scalarize(sum((probsys.veh.pos .* pos_scale/100).^2) + ((sum((vel_scale[1:3] .* probsys.veh.v[1:3]).^2))) + sum((probsys.veh.ω) .^2) + sum((probsys.veh.R .* R_scale) .^2)),
        (tsys) -> begin 
            get_pos = getu(tsys, tsys.model.veh.pos)
            get_omega = getu(tsys, tsys.model.veh.ω)
            get_v = getu(tsys, tsys.model.veh.v)
            get_R = getu(tsys, tsys.model.veh.R)
            get_omega_max = getp(tsys, tsys.ωmax)
            get_omega_weight = getp(tsys, tsys.sωmax)
            get_x = getp(tsys, tsys.model.inputx.vals)
            get_y = getp(tsys, tsys.model.inputy.vals)
            get_z = getp(tsys, tsys.model.inputz.vals)
            get_gimbal = getp(tsys, tsys.model.gimbal_angle)
            custfun = cvx_mod(tsys)
            nunk = length(unknowns(tsys))
            return function (model, δx, xref, symbolic_params, objexp)
                x_ctrl = get_x(symbolic_params)
                y_ctrl = get_y(symbolic_params)
                z_ctrl = get_z(symbolic_params)
                gimbal_angle = get_gimbal(symbolic_params)
                for (x,y,z) in Iterators.zip(x_ctrl, y_ctrl, z_ctrl)
                    # 10.5 degree tvc
                    @constraint(model, [z * tand(gimbal_angle), x, y] in SecondOrderCone())
                    # max throttle
                    @constraint(model, [1.0, x, y, z] in SecondOrderCone())
                end

                symbolic_state = δx .+ xref
                if get_omega_weight(symbolic_params) > 0.0
                    for x in eachcol(symbolic_state)
                        @constraint(model, [deg2rad(get_omega_max(symbolic_params)); get_omega(x)] in SecondOrderCone())
                    end
                end
                for x in eachcol(symbolic_state)
                    pos = get_pos(symbolic_state)
                    @constraint(model, [tand(60)*pos[3], pos[1], pos[2]] in SecondOrderCone())
                end

                final_dx = δx[:,end]
                final_xref = xref[:, end]
                tcvs = [get_omega(final_dx); get_R(final_dx); get_v(final_dx); get_pos(final_dx)]
                tcref = [get_omega(final_xref); get_R(final_xref); get_v(final_xref); get_pos(final_xref)]
                @variable(model, wfin[1:length(tcvs)])
                MOI.add_constraint(model.moi_backend, MOI.VectorAffineFunction([addvec2terms(tcvs); addvec2terms(wfin)], tcref), MOI.Zeros(length(tcvs)))
                #=
                final_symbolic_state = symbolic_state[:,end] .+ wfin
                @constraint(model, get_omega(final_symbolic_state) .== [0.0,0.0]) # omega
                @constraint(model, get_R(final_symbolic_state) .== [0.0,0.0]) # R
                @constraint(model, get_v(final_symbolic_state) .== [0.0,0.0,0.0]) # v
                @constraint(model, get_pos(final_symbolic_state) .== [0.0,0,0.0]) # pos
                =#

                return custfun(model, δx, xref, symbolic_params, objexp + 0.5*sum(wfin.*wfin)*1000000)
            end
        end, 
        (sys, l, y) -> begin 
            return custsys(sys,l,y)
        end)
end