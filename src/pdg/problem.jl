

function mkinit_interp(var, N, sol)
    sol(LinRange(0.0,1.0,N), idxs=var).u
end
mkinit(var, N, sol) = eachrow(stack(mkinit_interp(var, N, sol)))
lin_range_vals(i,f,n) = eachrow(reduce(hcat, LinRange(i, f, n)))
function descentproblem(probsys, sol, solsys; cvx_mod=s->((_, _, _, _, p)->p), custsys=(s,l,y)->s)
    vel_scale = sol.ps[solsys.veh.ρv]
    pos_scale = sol.ps[solsys.veh.ρpos]
    R_scale = sol.ps[solsys.veh.ρR]
    return trajopt(probsys, (0.0, 1.0), 41, 
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
        (probsys.veh.τa + probsys.veh.τp)/5 + sqrt_smooth(Symbolics.scalarize(sum(probsys.veh.u .^ 2))), 0,# -100*dot(probsys.veh.pos .* pos_scale, [1.0,0.0,0.0]), 
        max(tanh(Symbolics.scalarize(norm(probsys.veh.v))) * (probsys.veh.alpha - 25.0), 0.0) +
        #norm(probsys.veh.ρv .* probsys.veh.v) * probsys.veh.alpha + 
        20*ifelse(t>0.5, max(0.5^2 - Symbolics.scalarize(sum(probsys.veh.u .^ 2)), 0.0), 0.0) + 
        10*lim_viol1(probsys.veh.ua[1], probsys.veh.mach, probsys.veh.alpha1, probsys.veh.alpha2) + 
        10*lim_viol2(probsys.veh.ua[2], probsys.veh.mach, probsys.veh.alpha1, probsys.veh.alpha2) +
        Symbolics.scalarize(sum(max.(abs.(probsys.veh.ω) .- 10.0 ,0.0))), 0.0, # todo: alpha_max_aero (probsys.veh.alpha - 25.0)/50 - need to do expanded dynamics for the pdg phase
        Symbolics.scalarize(sum((probsys.veh.pos .* pos_scale/100).^2) + ((sum((vel_scale[1:3] .* probsys.veh.v[1:3]).^2))) + sum((probsys.veh.ω) .^2) + sum((probsys.veh.R .* R_scale .- R_final) .^2)),
        (tsys) -> begin 
            get_x = getp(tsys, tsys.model.inputx.vals)
            get_y = getp(tsys, tsys.model.inputy.vals)
            get_z = getp(tsys, tsys.model.inputz.vals)
            custfun = cvx_mod(tsys)
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
                return custfun(model, δx, xref, symbolic_params, objexp)
            end
        end, 
        (sys, l, y) -> begin 
            return custsys(sys,l,y)
        end)
end