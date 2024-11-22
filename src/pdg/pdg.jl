include("dynamics.jl")
include("lut_ship.jl")
include("problem.jl")

using Quickhull
using GeometryBasics
using StatsBase

probsys = build_example_problem()
ssys = structural_simplify(probsys)

tf_max = 15.0
tf_min = 0.25
pos_init = [500.0,40000.0,40000.0]
vel_init = [0,-750,-500]
R_init = [-deg2rad(atand(750,500)),0]
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
    ssys.inputz.vals => 0.0 * ones(20)
])
sol = solve(prob, Tsit5(); dtmax=0.0001)

prb = descentproblem(probsys, sol, ssys);
ui,xi,_,_,_,_,_,unk,_ = do_trajopt(prb; maxsteps=1);
u,x,wh,ch,rch,dlh,lnz,unk,tp = do_trajopt(prb; maxsteps=100);

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


prb_divert = descentproblem(probsys, sol_ws, ssys;
    cvx_mod = (tsys) -> begin 
        get_push_dir = getp(tsys, tsys.push_dir)
        get_ignpt = getp(tsys, tsys.ignpt)
        get_pos = getu(tsys, tsys.model.veh.pos)
        return function (model, δx, xref, symbolic_params, objexp)
            # only the tunables in symbolic params are actually symbolic; the rest are just numbers
            push_dir = get_push_dir(symbolic_params)
            ignpt = get_ignpt(symbolic_params)

            JuMP.@variable(model, c)
            JuMP.@variable(model, wc[1:3])
            pos = get_pos(δx[:, 21] .+ xref[:, 21])
            cis = @constraint(model, pos .+ wc .== ignpt .+ c .*push_dir)
            cvx_cst_est = () -> begin 
                res = -100*dot(value.(get_pos(δx[:, 21] .+ xref[:, 21])) - ignpt, push_dir)
                return res
            end
            (nst,npt) = size(δx)
            nonlin_cst = (res) -> begin 
                outp = -100*dot(get_pos(reshape(res[1:end-1], (nst, npt-1))[:,20]) - ignpt, push_dir)
                return outp
            end
            function postsolve(model)
            end
            return objexp + 100000*sum(wc.*wc) - 100*dot(get_pos(δx[:, 21] .+ xref[:, 21]) - ignpt, push_dir), cvx_cst_est, nonlin_cst, postsolve
        end
    end,
    custsys=(sys,l,y) -> begin 
        @parameters push_dir[1:3]=[1,0,0], [tunable=false,input=true]
        @parameters ignpt[1:3]=[0,0,0], [tunable=false,input=true]
        function add_divert!(i, u, p, c)
            mod = 100*norm([i.u[u.posx], i.u[u.posy], i.u[u.posz]] .- i.ps[p.ignpt])
           # i[u.l] = 5-mod
        end
        @named augmenting=ODESystem(Equation[], t, [], [push_dir,ignpt]; discrete_events=[
            [0.5] => (add_divert!, [l => :l, probsys.veh.pos[1] => :posx, probsys.veh.pos[2] => :posy, probsys.veh.pos[3] => :posz], [push_dir,ignpt], [], nothing)
        ])
        return extend(sys, augmenting)
    end
);

function propagate_sol(u)
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
    return solve(prob_res, Tsit5())
end

    upd_pushdir = setp(prb_divert[:tsys], prb_divert[:tsys].push_dir)
    upd_ignpt = setp(prb_divert[:tsys], prb_divert[:tsys].ignpt)
    upd_fuel_wt = setp(prb_divert[:tsys], prb_divert[:tsys].obj_weight_fuel)

    upd_pushdir(prb_divert[:pars], [0.0,0.0,1.0])
    upd_ignpt(prb_divert[:pars], ignpt)
    upd_fuel_wt(prb_divert[:pars], 0.0)
    
    ui,xi,_,_,_,_,_,unk,_ = do_trajopt(prb_divert; maxsteps=1);
    up,xp,whp,chp,rchp,dlhp,lnzp,unkp,tpp = do_trajopt(prb_divert; maxsteps=50, r=16);

    dirs = []
    pushed = Vector{Float64}[]

    rejected = 0
    errors = 0
    ph = nothing
    for i=1:100
        println("====== $i $i $i $i $i ======")
        dir_rand = rand(3) - [0.5, 0.5, 0.5]
        dir_rand = dir_rand/norm(dir_rand)
        upd_pushdir(prb_divert[:pars], dir_rand)
        upd_ignpt(prb_divert[:pars], if !isempty(pushed) && length(pushed) > 4
            ph = quickhull(convert(Vector{Vector{Float64}}, pushed))
            sample(ph.pts)
        else 
            ignpt
        end)
        try
            up,xp,whp,chp,rchp,dlhp,lnzp,unkp,tpp = do_trajopt(prb_divert; maxsteps=50, r=16);
            propagated = propagate_sol(up)
            if norm(propagated[ssys.veh.pos .* ssys.veh.ρpos][end]) > 20 || maximum(abs.(wh[end])) > 1e-3
                println("SOLN REJECT > tol")
                rejected += 1
                continue 
            end
            
            push!(pushed, xp[end][11:13,21])
            push!(dirs, dir_rand)
        catch e 
            println("caught error, continuing")
            errors += 1
        end
    end
    ph = quickhull(convert(Vector{Vector{Float64}}, pushed))
    using GLMakie
    Makie.lines(Point3.(sol_ws[ssys.veh.pos]))
    Makie.wireframe!(GeometryBasics.Mesh(GeometryBasics.Point3.(ph.pts), facets(ph)), color=:blue)
    scatter!(Point3.(pushed))
    scatter!(Point3.([xp[end][end-2:end,21]]), color=:red)

sol_res = propagate_sol(u)
sol_res = propagate_sol(up)

    #GLMakie.activate!()
    import CairoMakie 
    CairoMakie.activate!()

using GLMakie
function plot_soln(sol_res)
    
    retimer(t) = 10*(min(t, 0.5) * sol_res.ps[ssys.veh.τa] + max(t - 0.5, 0) * sol_res.ps[ssys.veh.τp])
    f=Figure(size=(1400,900))
    a=Axis3(f[1:3,1], aspect=:data, azimuth = -0.65π, xlabel="N (m)", ylabel="E (m)", zlabel="U (m)")
    Makie.lines!(a,Point3.(sol_res(LinRange(0.0,0.5,500), idxs = ssys.veh.pos .* ssys.veh.ρpos).u))
    Makie.lines!(a,Point3.(sol_res(LinRange(0.5,1.0,500), idxs = ssys.veh.pos .* ssys.veh.ρpos).u))

    naxes = 30
    for rp in zip(
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.pos .* ssys.veh.ρpos).u),
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.pos .* ssys.veh.ρpos .+ rquat(ssys.veh.R) * [0,0,1000]).u))
        Makie.lines!(a, [rp[1], rp[2]], color=:blue)
    end

    for rp in zip(
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.pos .* ssys.veh.ρpos).u),
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.pos .* ssys.veh.ρpos .+ rquat(ssys.veh.R) * [0,1000,0]).u))
        Makie.lines!(a, [rp[1], rp[2]], color=:green)
    end

    for rp in zip(
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.pos .* ssys.veh.ρpos).u),
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.pos .* ssys.veh.ρpos .+ rquat(ssys.veh.R) * [1000,0,0]).u))
        Makie.lines!(a, [rp[1], rp[2]], color=:red)
    end

    b1 = Makie.Axis(f[1, 2], title="Position")
    b2 = Makie.Axis(f[1, 2], yaxisposition = :right)
    unpowered_pos = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.pos .* ssys.veh.ρpos)
    powered_pos = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.pos .* ssys.veh.ρpos)
    zm = Makie.lines!(b1, retimer.(unpowered_pos.t), getindex.(unpowered_pos.u, 1), label="N(m)", color=:red)
    xm = Makie.lines!(b2, retimer.(unpowered_pos.t), getindex.(unpowered_pos.u, 2), label="E(m)", color=:green)
    ym = Makie.lines!(b2, retimer.(unpowered_pos.t), getindex.(unpowered_pos.u, 3), label="U(m)", color=:blue)
    zam = Makie.lines!(b1, retimer.(powered_pos.t), getindex.(powered_pos.u, 1), label="N(m)", color=:red, linestyle=:dot)
    xam = Makie.lines!(b2, retimer.(powered_pos.t), getindex.(powered_pos.u, 2), label="E(m)", color=:green, linestyle=:dot)
    yam = Makie.lines!(b2, retimer.(powered_pos.t), getindex.(powered_pos.u, 3), label="U(m)", color=:blue, linestyle=:dot)
    Legend(f[1,2], [zm, xm, ym], ["N(m)", "E(m)", "U(m)"], "Axis",
        tellheight = false,
        tellwidth = false,
        margin = (5,5,5,5),
        halign = :right, 
        valign = :top)
    vi = [9, 10, 11]
    b3 = Makie.Axis(f[2, 2], title="U Velocity (m/s)")
    b4 = Makie.Axis(f[3, 2], title="N-E Velocity (m/s)", xlabel="Time (s)")
    unpowered_vel = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.v .* ssys.veh.ρv)
    powered_vel = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.v .* ssys.veh.ρv)
    zvm = Makie.lines!(b4, retimer.(unpowered_vel.t), getindex.(unpowered_vel.u, 1), color=:red)
    xvm = Makie.lines!(b4, retimer.(unpowered_vel.t), getindex.(unpowered_vel.u, 2), color=:green)
    yvm = Makie.lines!(b3, retimer.(unpowered_vel.t), getindex.(unpowered_vel.u, 3), color=:blue)
    zvam = Makie.lines!(b4, retimer.(powered_vel.t), getindex.(powered_vel.u, 1), color=:red, linestyle=:dot)
    xvam = Makie.lines!(b4, retimer.(powered_vel.t), getindex.(powered_vel.u, 2), color=:green, linestyle=:dot)
    yvam = Makie.lines!(b3, retimer.(powered_vel.t), getindex.(powered_vel.u, 3), color=:blue, linestyle=:dot)


    Legend(f[2,2], [yvam, yvm], ["powered", "aerodynamic"], "Phase",
        tellheight = false,
        tellwidth = false,
        margin = (5,5,5,5),
        halign = :left, 
        valign = :top)

    b5 = Makie.Axis(f[1, 3], title="AoA (°)", limits=(nothing, (0.0,30.0)))
    aoa_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.alpha)
    aoa_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.alpha)
    Makie.lines!(b5, retimer.(unpowered_vel.t), aoa_unpowered.u, color="#56B4E9")
    Makie.lines!(b5, retimer.(powered_vel.t), aoa_powered.u, color="#56B4E9",linestyle=:dot)

    b5 = Makie.Axis(f[2, 3], title="AoA 1/2 (°)", limits=(nothing, (-30.0,30.0)))
    α1_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.alpha1)
    α1_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.alpha1)
    Makie.lines!(b5, retimer.(unpowered_vel.t), α1_unpowered.u, color="#E956B4")
    Makie.lines!(b5, retimer.(powered_vel.t), α1_powered.u, color="#E956B4",linestyle=:dot)
    α2_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.alpha2)
    α2_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.alpha2)
    Makie.lines!(b5, retimer.(unpowered_vel.t), α2_unpowered.u, color="#B4E956")
    Makie.lines!(b5, retimer.(powered_vel.t), α2_powered.u, color="#B4E956",linestyle=:dot)

    b6 = Makie.Axis(f[3, 3], title="Omega (°/s)", xlabel="Time (s)")
    ω_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.ω)
    ω_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.ω)
    zvm = Makie.lines!(b6, retimer.(unpowered_vel.t), rad2deg.(getindex.(ω_unpowered.u, 1)), color=:red)
    xvm = Makie.lines!(b6, retimer.(unpowered_vel.t), rad2deg.(getindex.(ω_unpowered.u, 2)), color=:green)
    zvam = Makie.lines!(b6, retimer.(powered_vel.t), rad2deg.(getindex.(ω_powered.u, 1)), color=:red, linestyle=:dot)
    xvam = Makie.lines!(b6, retimer.(powered_vel.t), rad2deg.(getindex.(ω_powered.u, 2)), color=:green, linestyle=:dot)

    b7 = Makie.Axis(f[1, 4], title="Lift command")
    ua_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.ua)
    ua_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.ua)
    Makie.lines!(b7, retimer.(unpowered_vel.t), getindex.(ua_unpowered.u, 1), color=:red)
    Makie.lines!(b7, retimer.(unpowered_vel.t), getindex.(ua_unpowered.u, 2), color=:green)
    Makie.lines!(b7, retimer.(ua_powered.t), getindex.(ua_powered.u, 1), color=:red)
    Makie.lines!(b7, retimer.(ua_powered.t), getindex.(ua_powered.u, 2), color=:green)

    atref = sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.mach).t
    ulim1 = upper_lim1_lut.(sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.mach), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha1), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha2))
    llim1 = lower_lim1_lut.(sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.mach), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha1), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha2))
    Makie.lines!(b7, retimer.(atref), collect(ulim1), color=:red, linestyle=:dash)
    Makie.lines!(b7, retimer.(atref), collect((-).(llim1)), color=:red, linestyle=:dash)
    ulim2 = upper_lim2_lut.(sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.mach), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha1), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha2))
    llim2 = lower_lim2_lut.(sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.mach), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha1), sol_res(LinRange(0.0,1.0,100), idxs=ssys.veh.alpha2))
    Makie.lines!(b7, retimer.(atref), collect(ulim2), color=:green, linestyle=:dash)
    Makie.lines!(b7, retimer.(atref), collect((-).(llim2)), color=:green, linestyle=:dash)

    b8 = Makie.Axis(f[2, 4], title="Fin forces (Wind, N)")
    clf_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=1000*ssys.veh.ρᵣ*ssys.veh.aero_ctrl_lift)
    clf_powered = sol_res(LinRange(0.5,1.0,100), idxs=1000*ssys.veh.ρᵣ*ssys.veh.aero_ctrl_lift)
    cdf_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=1000*ssys.veh.ρᵣ*ssys.veh.aero_ctrl_drag)
    cdf_powered = sol_res(LinRange(0.5,1.0,100), idxs=1000*ssys.veh.ρᵣ*ssys.veh.aero_ctrl_drag)
    zvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(clf_unpowered.u, 1), color=:red)
    xvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(clf_unpowered.u, 2), color=:green)
    xvm = Makie.lines!(b8, retimer.(unpowered_vel.t), cdf_unpowered.u, color=:blue)
    zvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(clf_powered.u, 1), color=:red, linestyle=:dot)
    xvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(clf_powered.u, 2), color=:green, linestyle=:dot)
    xvm = Makie.lines!(b8, retimer.(powered_vel.t), cdf_powered.u, color=:blue, linestyle=:dot)

    b8 = Makie.Axis(f[3, 4], title="Aerodynamic body forces (Wind, N)", xlabel="Time (s)")
    cbf_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.aero_force .- ssys.veh.ρᵣ *ssys.veh.aero_ctrl_force)
    cbf_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.aero_force .- ssys.veh.ρᵣ *ssys.veh.aero_ctrl_force)
    zvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(cbf_unpowered.u, 1), color=:red)
    xvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(cbf_unpowered.u, 2), color=:green)
    yvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(cbf_unpowered.u, 3), color=:blue)
    zvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(cbf_powered.u, 1), color=:red, linestyle=:dot)
    xvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(cbf_powered.u, 2), color=:green, linestyle=:dot)
    yvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(cbf_powered.u, 3), color=:blue, linestyle=:dot)

    b9 = Makie.Axis(f[1, 5], title="Norm thrust (% of max)", limits=(nothing, (0.0,1.2)))
    u_powered = sol_res(LinRange(0.5,1.0,100), idxs=Symbolics.scalarize(norm(ssys.veh.u)))
    zvm = Makie.lines!(b9, retimer.(powered_vel.t), getindex.(u_powered.u, 1))

    b10 = Makie.Axis(f[2, 5], title="Acceleration from thrust (Body, m/s^2)")
    th_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.th/(ssys.veh.m*ssys.veh.mdry))
    zvm = Makie.lines!(b10, retimer.(powered_vel.t), getindex.(th_powered.u, 1), color=:red)
    xvm = Makie.lines!(b10, retimer.(powered_vel.t), getindex.(th_powered.u, 2), color=:green)
    yvm = Makie.lines!(b10, retimer.(powered_vel.t), getindex.(th_powered.u, 3), color=:blue)

    b11 = Makie.Axis(f[3, 5], title="Fuel Mass (kg)", limits=(nothing,(0.0,12000)), xlabel="Time (s)")
    m_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.m*ssys.veh.mdry - ssys.veh.mdry)
    zvm = Makie.lines!(b11, retimer.(powered_vel.t), m_powered.u)
    f
end
import CairoMakie
save("sim.pdf", f; backend=CairoMakie, size=(1200,900))


Makie.lines(sol[sim.iv], norm.(sol[sim.alpha]))

