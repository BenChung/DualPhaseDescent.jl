include("dynamics.jl")
include("lut_ship.jl")
include("problem.jl")

using Quickhull
using GeometryBasics
using StatsBase
using Printf
using TickTock
using Polyhedra

probsys = build_example_problem()
ssys = structural_simplify(probsys)

function solve_single_shot(
    pos_init = [500.0,2500.0,15000.0],
    vel_init = [0,-150,-350],
    R_init = [-deg2rad(atand(150,350)),0],
    ω_init = [0,0],
    m_init = 19516.0)

    pos_final = [0,0,0.0]
    vel_final = [0,0,0.0]
    R_final = [0.0,0]
    ω_final = [0,0.0]

    R_scale = [1.0,1.0]
    pos_scale = [10000.0,10000.0,10000.0]
    vel_scale = [100.0,100.0,100.0]
    ω_scale = [1.0,1.0]
    m_scale = 10516.0
    prob = ODEProblem(ssys, [
        ssys.veh.m => m_init / m_scale
        ssys.veh.ω => ω_init
        ssys.veh.R => R_init ./ R_scale
        ssys.veh.pos => pos_init ./ pos_scale
        ssys.veh.v => vel_init ./ vel_scale
        ssys.veh.τa => 40.0/10
        ssys.veh.τp => 40.0/10
    ], (0.0, 1.0), [
        ssys.veh.ρv => vel_scale
        ssys.veh.ρpos => pos_scale
        ssys.veh.ρm => m_scale
        ssys.input_fin1.vals => 0.0*ones(20)
        ssys.input_fin2.vals => -0.0*ones(20) # 0.0*ones(20) #
        ssys.inputz.vals => 0.0 * ones(20)
    ])
    sol = solve(prob, Tsit5(); dtmax=0.0001)

    prb = descentproblem(probsys, sol, ssys);


    setp(prb[:tsys], prb[:tsys].obj_weight_fuel)(prb[:pars], 1.0)
    setp(prb[:tsys], prb[:tsys].obj_weight_time)(prb[:pars], 0.0)
    setp(prb[:tsys], prb[:tsys].obj_weight_ω)(prb[:pars], 0.0)

    #setp(prb[:tsys], prb[:tsys].obj_weight_fuel)(prb[:pars], 0.0)
    #setp(prb[:tsys], prb[:tsys].obj_weight_time)(prb[:pars], 0.05)

    setp(prb[:tsys], prb[:tsys].sth)(prb[:pars], 100.0)
    setp(prb[:tsys], prb[:tsys].ωmax)(prb[:pars], 10.0)
    setp(prb[:tsys], prb[:tsys].sωmax)(prb[:pars], 0.1)
    setp(prb[:tsys], prb[:tsys].sqmax)(prb[:pars], 5e-4)
    setp(prb[:tsys], prb[:tsys].sqαmax)(prb[:pars], 1e-6)
    setp(prb[:tsys], prb[:tsys].qmax)(prb[:pars], 8e4) # 80kPa, from real Falcon trajes
    setp(prb[:tsys], prb[:tsys].qαmax)(prb[:pars], 1e6)

    u,x,wh,ch,rch,dlh,lnz,unk,tp = do_trajopt(prb; maxsteps=100,tol=1e-3,r=1);
    return u,x,wh,ch,rch,dlh,lnz,unk,tp
end


function propagate_sol(ssys, u, tp; 
    pos_init = [500.0,2500.0,15000.0],
    vel_init = [0,-150,-350],
    R_init = [-deg2rad(atand(150,350)),0],
    ω_init = [0,0],
    m_init = 19516.0,
    veh=ssys.veh)
    pos_final = [0,0,0.0]
    vel_final = [0,0,0.0]
    R_final = [0.0,0]
    ω_final = [0,0.0]

    R_scale = [1.0,1.0]
    pos_scale = [10000.0,10000.0,10000.0]
    vel_scale = [100.0,100.0,100.0]
    ω_scale = [1.0,1.0]
    m_scale = 10516.0
    prob_res = ODEProblem(ssys, [
        veh.m => m_init / m_scale
        veh.ω => ω_init
        veh.R => R_init
        veh.pos => pos_init ./ pos_scale
        veh.v => vel_init ./ vel_scale
    ], (0.0, 1.0), [
        veh.ρv => vel_scale;
        veh.ρpos => pos_scale;
        veh.ρm => m_scale;
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

function dual_phase_reachability(x, u, tp, niters=10000)
    sol_ws = propagate_sol(ssys, u, tp)    
    ignpt = sol_ws(0.5, idxs=ssys.veh.pos)
    prb_divert = descentproblem(probsys, sol_ws, ssys;
        cvx_mod = (tsys) -> begin 
            get_push_dir = getp(tsys, tsys.push_dir)
            get_ignpt = getp(tsys, tsys.sym_ignpt)
            get_pos = getu(tsys, tsys.model.veh.pos)
            return function (model, δx, xref, symbolic_params, objexp)
                # only the tunables in symbolic params are actually symbolic; the rest are just numbers
                push_dir = get_push_dir(symbolic_params)
                sym_ignpt = get_ignpt(symbolic_params)

                JuMP.@variable(model, c)
                JuMP.@variable(model, wc[1:3])
                pos = get_pos(δx[:, 21] .+ xref[:, 21])
                cis = @constraint(model, pos .+ wc .== sym_ignpt .+ c .*push_dir)
                cvx_cst_est = () -> begin 
                    res = -100*dot(value.(get_pos(δx[:, 21] .+ xref[:, 21])) - sym_ignpt, push_dir)
                    return res
                end
                (nst,npt) = size(δx)
                nonlin_cst = (res) -> begin 
                    outp = -100*dot(get_pos(reshape(res[1:end-1], (nst, npt-1))[:,20]) - sym_ignpt, push_dir)
                    return outp
                end
                function postsolve(model)
                end
                return objexp + 1000*sum(wc.*wc) - 100*dot(get_pos(δx[:, 21] .+ xref[:, 21]) - sym_ignpt, push_dir), cvx_cst_est, nonlin_cst, postsolve
            end
        end,
        custsys=(sys,l,y) -> begin 
            @parameters push_dir[1:3]=[1,0,0], [tunable=false,input=true]
            @parameters sym_ignpt[1:3]=[0,0,0], [tunable=false,input=true]
            function add_divert!(i, u, p, c)
                mod = 100*norm([i.u[u.posx], i.u[u.posy], i.u[u.posz]] .- i.ps[p.sym_ignpt])
            # i[u.l] = 5-mod
            end
            @named augmenting=ODESystem(Equation[], t, [], [push_dir,sym_ignpt]; discrete_events=[
                [0.5] => (add_divert!, [l => :l, probsys.veh.pos[1] => :posx, probsys.veh.pos[2] => :posy, probsys.veh.pos[3] => :posz], [push_dir,sym_ignpt], [], nothing)
            ])
            return extend(sys, augmenting)
        end
    );


    upd_pushdir = setp(prb_divert[:tsys], prb_divert[:tsys].push_dir)
    upd_ignpt = setp(prb_divert[:tsys], prb_divert[:tsys].sym_ignpt)
    upd_fuel_wt = setp(prb_divert[:tsys], prb_divert[:tsys].obj_weight_fuel)

    upd_pushdir(prb_divert[:pars], [1.0,0.0,0.0])
    upd_ignpt(prb_divert[:pars], ignpt)
    upd_fuel_wt(prb_divert[:pars], 0.0)
    setp(prb_divert[:tsys], prb_divert[:tsys].obj_weight_time)(prb_divert[:pars], 0.0)
    setp(prb_divert[:tsys], prb_divert[:tsys].obj_weight_ω)(prb_divert[:pars], 0.0)

    
    setp(prb_divert[:tsys], prb_divert[:tsys].ωmax)(prb_divert[:pars], 10.0)
    setp(prb_divert[:tsys], prb_divert[:tsys].sωmax)(prb_divert[:pars], 0.1)
    setp(prb_divert[:tsys], prb_divert[:tsys].sqmax)(prb_divert[:pars], 5e-4)
    setp(prb_divert[:tsys], prb_divert[:tsys].sqαmax)(prb_divert[:pars], 1e-5)
    setp(prb_divert[:tsys], prb_divert[:tsys].qmax)(prb_divert[:pars], 8e4) # 80kPa, from real Falcon trajes
    setp(prb_divert[:tsys], prb_divert[:tsys].qαmax)(prb_divert[:pars], 1e6)
    
    ifun = (_) -> x[end]
    up,xp,whp,chp,rchp,dlhp,lnzp,unkp,tpp = do_trajopt(prb_divert; maxsteps=50, r=1, tol=1e-3, 
        initfun=ifun, uguess=u[end]);

    function do_reachability_problem(prb_divert, ignpt, u, trajes=1)
        dirs = Vector{Float64}[]
        pushed = Pair{Vector{Float64}, Tuple{Matrix{Float64}, Vector{Float64}}}[ignpt => (x[end],u[end])]
        chhists = Vector{Float64}[]
        src = [-1]
        times = []

        rejects = []
        rejected = 0
        errors = 0
        ph = nothing
        tick()
        for i=1:trajes
            println("====== $i $i $i $i $i $i $i $i $i $i $i $i $i $i $i ======")
            dir_rand = rand(3) - [0.5, 0.5, 0.5]
            dir_rand = dir_rand/norm(dir_rand)
            upd_pushdir(prb_divert[:pars], dir_rand)
            control_guess = nothing
            src_ind = 0
            upd_ignpt(prb_divert[:pars], if !isempty(pushed) && length(pushed) > 4
                ph = quickhull(convert(Vector{Vector{Float64}}, first.(pushed)))
                result = sample(ph.pts)
                src_ind = findfirst(pr -> pr[1] ≈ result, pushed)
                (state_guess, control_guess) = last(pushed[src_ind])
                result
            else 
                src_ind = 1
                state_guess = x[end]
                control_guess = u[end]
                ignpt
            end)
            try
                ifun = (_) -> x[end]
                up,xp,whp,chp,rchp,dlhp,lnzp,unkp,tpp = do_trajopt(prb_divert; maxsteps=40, r=16, tol=1e-3,
                    initfun=(prb) -> state_guess);
                propagated = propagate_sol(ssys, up, tpp)
                if norm(propagated[ssys.veh.posp][end]) > 20 || maximum(abs.(whp[end])) > 1e-3
                    println("SOLN REJECT > tol")
                    rejected += 1
                    push!(rejects, (up[end], whp[end]))
                    continue 
                end
                
                push!(pushed, get_pos(xp[end][:,21]) => (xp[end], up[end]))
                push!(src, src_ind)
                push!(chhists, rchp)
                push!(dirs, dir_rand)
                push!(times, peektimer())
            catch e 
                println("caught error, continuing")
                if isa(e, InterruptException)
                    println("it was an interrupt")
                    break
                end
                errors += 1
            end
        end
        tock()
        return rejects, pushed, src, chhists, dirs, times, errors, rejected
    end
    rejects, pushed, src, chhists, dirs, times, errors, rejected = do_reachability_problem(prb_divert, ignpt, u, niters);
    ph = quickhull(convert(Vector{Vector{Float64}}, first.(pushed)))
    return rejects, pushed, src, chhists, dirs, times, errors, rejected, ph
end
#=
    using CairoMakie
    include("trajplots.jl")

    save("spbm_convplot.pdf", spbm_convplot(chhists); size=(400,400))

    save("reachable.pdf", plot_polytope(sol_ws, pushed, ph); backend=CairoMakie, size=(900,900))

    save("running_time_dist.pdf", hist(times[2:end] - times[1:end-1], bins=200, axis=(xticks=0:9,xlabel="Running time (s)", ylabel="Subproblems")); size=(400,400))
    g = make_convergence_plot(sol_ws, pushed)

    save("expansion_trajectories.pdf", expansion_trajectories(pushed, sol_ws, ph); size=(400,400))
    save("reachability-convergence.pdf", g; size=(400,400))




sol_res = propagate_sol(ssys, up)
sol_res = propagate_sol(ssys, [pushed[461][2]])
f = plot_soln(sol_res)

save("sim.pdf", f; backend=CairoMakie, size=(1200,900))

    #GLMakie.activate!()
    using CairoMakie 
    CairoMakie.activate!()

using GLMakie
import CairoMakie
save("sim.pdf", f; backend=CairoMakie, size=(1200,900))


Makie.lines(sol[sim.iv], norm.(sol[sim.alpha]))

=#