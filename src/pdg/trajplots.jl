
function calc_volume_convergence(sol_ws, pushed)
    scalefactor = sol_ws.ps[ssys.veh.ρpos]
    pts = [scalefactor .* p for p in first.(pushed)]
    vols = []
    for i = 4:maximum(eachindex(pushed))
        coords = pts[1:i]
        ph = quickhull(convert(Vector{Vector{Float64}}, coords))
        push!(vols, Polyhedra.volume(polyhedron(vrep(ph.pts[vertices(ph)]))))
        if i % 100 == 0
            @show i
        end
    end
    return vols
end

function make_convergence_plot(sol_ws, pushed) # danger! slow!
    
    vc = calc_volume_convergence(sol_ws, pushed)

    function makelab(time_in_s)
        if floor(Int, modf(time_in_s/60/60)[1]*60) > 0
            return "$(floor(Int, time_in_s/60/60)) hr, $(floor(Int, modf(time_in_s/60/60)[1]*60)) m"
        else 
            return "$(floor(Int, time_in_s/60/60)) hr"
        end 
    end
    f = Figure()
    timeticks = [60 * 60 .*[0,1,2,3]; maximum(times)]
    ax = Makie.Axis(f[1,1], xticks = timeticks, 
        xtickformat = values -> makelab.(values), xlabel="Wall clock time", ylabel="Polytope volume (m^3)")
    Makie.lines!(ax, times[3:end], vc)
    display(f)
    indexlims = (1-length(vc)*0.05,length(vc)+length(vc)*0.1)
    tlims = ax.xaxis.attributes.limits.val
    calcticks = (indexlims[2] - indexlims[1]) .* (timeticks .- tlims[1])./(tlims[2] - tlims[1]) .+ indexlims[1]
    ax2 = Makie.Axis(f[1, 1], xaxisposition = :top, xlabel="Iteration number",
        limits=(indexlims,nothing), xticks=calcticks,
        xtickformat = ticks -> string.(Int.(round.(ticks/100)*100)))
    hidespines!(ax2)
    hideydecorations!(ax2)
    return f 
    #save("reachability-convergence.pdf", f; size=(400,400))
end

function spbm_convplot(chhists)
    
    hist_lens = length.(chhists)
    f=Figure()
    ax=Makie.Axis(f[1,1], 
        xlabel="Subproblem iteration count", 
        ylabel="Converged in < iters",
        ytickformat=labs->["$(round(Int, l*100))%" for l in labs],
        xticks=0:5:40)
    Makie.lines!(ax,[(length(hist_lens) - sum(hist_lens .> i))/(length(hist_lens)+rejected+errors) for i=1:40])
    return f
end

function plot_polytope(sol_ws, pushed, ph)
    scalefactor = sol_ws.ps[ssys.veh.ρpos]
    polypts = [scalefactor .* p for p in ph.pts[ph.vertices]]
    
    #ax = Axis3(f[1:3,1], aspect=:data)
    #Makie.lines!(ax,Point3.(sol_ws[ssys.veh.pos]))
    
    #scatter!(ax, Point3.(first.(pushed)))
    #scatter!(Point3.([xp[end][end-2:end,21]]), color=:red)

    maxN = findmax(p->p[1][1], pushed)
    minN = findmin(p->p[1][1], pushed)
    maxE = findmax(p->p[1][2], pushed)
    minE = findmin(p->p[1][2], pushed)
    maxU = findmax(p->p[1][3], pushed)
    minU = findmin(p->p[1][3], pushed)
    sols = propagate_sol.((ssys,), map(x->[last(x)[2]], getindex.((pushed,), last.([maxN, minN, maxE, minE, maxU, minU]))));

    function plot_onto_axis(ax, idx, sols)
        for soln in sols 
            pts = soln(LinRange(0.0,0.5,500), idxs = Symbolics.scalarize(ssys.veh.posp)[idx]).u
            Makie.lines!(ax, Point2.(pts), color=:blue)
            pts = soln(LinRange(0.5,1.0,500), idxs = Symbolics.scalarize(ssys.veh.posp)[idx]).u
            Makie.lines!(ax, Point2.(pts), color=:red)
        end
        
        projhull = quickhull(getindex.(polypts, (idx,)))
        bdry = [(Point2(projhull.pts[pr[1]]), Point2(projhull.pts[pr[2]])) for pr in facets(projhull)]
        Makie.linesegments!(ax, bdry)
    end

    kmformat = values -> [@sprintf("%.1f", value/1000) for value in values]
    f = Figure(size=(900,900),figure_padding=50)
    a=Axis3(f[1:3,1], aspect=:data, azimuth = -0.65π, 
        xlabel="N (km)", ylabel="E (km)", zlabel="U (km)",
        xtickformat = kmformat,
        ytickformat = kmformat,
        ztickformat = kmformat)
    for soln in sols 
        pts = soln(LinRange(0.0,0.5,500), idxs = Symbolics.scalarize(ssys.veh.posp)).u
        Makie.lines!(a, Point3.(pts), color=:blue)
        pts = soln(LinRange(0.5,1.0,500), idxs = Symbolics.scalarize(ssys.veh.posp)).u
        Makie.lines!(a, Point3.(pts), color=:red)
    end
    Makie.wireframe!(a,GeometryBasics.Mesh(GeometryBasics.Point3.([scalefactor .* p for p in ph.pts]), facets(ph)))    
    ax = Makie.Axis(f[1:3,2], xtickformat = kmformat, ytickformat = kmformat, xlabel="N (km)", ylabel="U (km)")
    plot_onto_axis(ax, [1,3], [sols[[1,2]]; sols[[5,6]]])
    ax = Makie.Axis(f[1:3,3], xtickformat = kmformat, ytickformat = kmformat, xlabel="E (km)")
    plot_onto_axis(ax, [2,3], sols[3:end])
    return f
end


function plot_soln(sol_res)

    unpowered_style = :dot 
    powered_style = :solid
    
    retimer(t) = 10*(min(t, 0.5) * sol_res.ps[ssys.veh.τa] + max(t - 0.5, 0) * sol_res.ps[ssys.veh.τp])
    f=Makie.Figure(size=(1400,900),figure_padding=50)
    kmformat = values -> [@sprintf("%.1f", value/1000) for value in values]
    a=Axis3(f[1:3,1], aspect=:data, azimuth = -0.65π, 
        xlabel="N (km)", ylabel="E (km)", zlabel="U (km)",
        xtickformat = kmformat,
        ytickformat = kmformat,
        ztickformat = kmformat)
    Makie.lines!(a,Point3.(sol_res(LinRange(0.0,0.5,500), idxs = ssys.veh.posp).u))
    Makie.lines!(a,Point3.(sol_res(LinRange(0.5,1.0,500), idxs = ssys.veh.posp).u))

    naxes = 30
    for rp in zip(
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.posp).u),
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.posp .+ rquat(ssys.veh.R) * [0,0,1000]).u))
        Makie.lines!(a, [rp[1], rp[2]], color=:blue)
    end

    for rp in zip(
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.posp).u),
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.posp .+ rquat(ssys.veh.R) * [0,1000,0]).u))
        Makie.lines!(a, [rp[1], rp[2]], color=:green)
    end

    for rp in zip(
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.posp).u),
        Point3.(sol_res(LinRange(0.0,1.0,naxes), idxs = ssys.veh.posp .+ rquat(ssys.veh.R) * [1000,0,0]).u))
        Makie.lines!(a, [rp[1], rp[2]], color=:red)
    end

    b1 = Makie.Axis(f[2, 2], title="N Position (m)")
    b2 = Makie.Axis(f[1, 2], title="E/U Position (m)")
    unpowered_pos = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.posp)
    powered_pos = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.posp)
    zm = Makie.lines!(b1, retimer.(unpowered_pos.t), getindex.(unpowered_pos.u, 1), label="N(m)", color=:red, linestyle=unpowered_style)
    xm = Makie.lines!(b2, retimer.(unpowered_pos.t), getindex.(unpowered_pos.u, 2), label="E(m)", color=:green, linestyle=unpowered_style)
    ym = Makie.lines!(b2, retimer.(unpowered_pos.t), getindex.(unpowered_pos.u, 3), label="U(m)", color=:blue, linestyle=unpowered_style)
    zam = Makie.lines!(b1, retimer.(powered_pos.t), getindex.(powered_pos.u, 1), label="N(m)", color=:red, linestyle=powered_style)
    xam = Makie.lines!(b2, retimer.(powered_pos.t), getindex.(powered_pos.u, 2), label="E(m)", color=:green, linestyle=powered_style)
    yam = Makie.lines!(b2, retimer.(powered_pos.t), getindex.(powered_pos.u, 3), label="U(m)", color=:blue, linestyle=powered_style)
    Legend(f[1,2], [zm, xm, ym], ["N(m)", "E(m)", "U(m)"], "Axis",
        tellheight = false,
        tellwidth = false,
        margin = (5,5,5,5),
        halign = :right, 
        valign = :top)
    vi = [9, 10, 11]
    b3 = Makie.Axis(f[3, 2], title="Velocity (m/s)", xlabel="Time (s)")
    unpowered_vel = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.vp)
    powered_vel = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.vp)
    zvm = Makie.lines!(b3, retimer.(unpowered_vel.t), getindex.(unpowered_vel.u, 1), color=:red, linestyle=unpowered_style)
    xvm = Makie.lines!(b3, retimer.(unpowered_vel.t), getindex.(unpowered_vel.u, 2), color=:green, linestyle=unpowered_style)
    yvm = Makie.lines!(b3, retimer.(unpowered_vel.t), getindex.(unpowered_vel.u, 3), color=:blue, linestyle=unpowered_style)
    zvam = Makie.lines!(b3, retimer.(powered_vel.t), getindex.(powered_vel.u, 1), color=:red, linestyle=powered_style)
    xvam = Makie.lines!(b3, retimer.(powered_vel.t), getindex.(powered_vel.u, 2), color=:green, linestyle=powered_style)
    yvam = Makie.lines!(b3, retimer.(powered_vel.t), getindex.(powered_vel.u, 3), color=:blue, linestyle=powered_style)


    Legend(f[2,2], [zm, zam], [ "aero", "powered"], "Phase",
        tellheight = false,
        tellwidth = false,
        margin = (5,5,5,5),
        halign = :left, 
        valign = :bottom)

    b5 = Makie.Axis(f[1, 3], title="AoA (°)", limits=(nothing, (0.0,30.0)))
    aoa_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.alpha)
    aoa_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.alpha)
    aoa_lim = sol_res(LinRange(0.0,1.0,200), idxs=1/tanh(Symbolics.scalarize(norm(ssys.veh.v)) + 1e-5) * (25.0))
    Makie.lines!(b5, retimer.(unpowered_vel.t), aoa_unpowered.u, color="#56B4E9", linestyle=unpowered_style)
    Makie.lines!(b5, retimer.(powered_vel.t), aoa_powered.u, color="#56B4E9",linestyle=powered_style)
    Makie.lines!(b5, retimer.(aoa_lim.t), aoa_lim.u, color=:red,linestyle=:dash)

    b5 = Makie.Axis(f[2, 3], title="Dynamic Pressure (Pa)")
    q_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.q)
    q_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.q)
    hlines!(b5, [8e4], color="#E956B4", linestyle=:dash)
    Makie.lines!(b5, retimer.(unpowered_vel.t), q_unpowered.u, color="#E956B4", linestyle=unpowered_style)
    Makie.lines!(b5, retimer.(powered_vel.t), q_powered.u, color="#E956B4",linestyle=powered_style)

    b6 = Makie.Axis(f[3, 3], title="||ω||₂ (°/s)", xlabel="Time (s)")
    ω_unpowered = sol_res(LinRange(0.0,0.5,10000), idxs=ssys.veh.ω)
    ω_powered = sol_res(LinRange(0.5,1.0,10000), idxs=ssys.veh.ω)
    hlines!(b6, [10.0], color=:blue, linestyle=:dash)
    zvm = Makie.lines!(b6, retimer.(ω_unpowered.t), rad2deg.(norm.(ω_unpowered.u)), color=:blue, linestyle=unpowered_style)
    zvam = Makie.lines!(b6, retimer.(ω_powered.t), rad2deg.(norm.(ω_powered.u)), linestyle=powered_style, color=:blue)

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

    b8 = Makie.Axis(f[2, 4], title="Aero accel. (Inertial, m/s^2)")
    caf_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.aero_force/ssys.veh.mp)
    caf_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.aero_force/ssys.veh.mp)
    zvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(caf_unpowered.u, 1), color=:red, linestyle=unpowered_style)
    xvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(caf_unpowered.u, 2), color=:green, linestyle=unpowered_style)
    xvm = Makie.lines!(b8, retimer.(unpowered_vel.t), getindex.(caf_unpowered.u, 3), color=:blue, linestyle=unpowered_style)
    zvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(caf_powered.u, 1), color=:red, linestyle=powered_style)
    xvam = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(caf_powered.u, 2), color=:green, linestyle=powered_style)
    xvm = Makie.lines!(b8, retimer.(powered_vel.t), getindex.(caf_powered.u, 3), color=:blue, linestyle=powered_style)

    b8 = Makie.Axis(f[3, 4], title="qα (Pa°)", xlabel="Time (s)")
    qα_unpowered = sol_res(LinRange(0.0,0.5,100), idxs=ssys.veh.q * ssys.veh.alpha)
    qα_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.q * ssys.veh.alpha)
    hlines!(b8, [1e6], color=:red, linestyle=:dash)
    zvm = Makie.lines!(b8, retimer.(unpowered_vel.t), qα_unpowered.u, color=:red, linestyle=unpowered_style)
    zvam = Makie.lines!(b8, retimer.(powered_vel.t), qα_powered.u, color=:red, linestyle=powered_style)

    b9 = Makie.Axis(f[1, 5], title="Norm thrust (% of max)", limits=(nothing, (0.0,1.2)))
    u_powered = sol_res(LinRange(0.5,1.0,100), idxs=Symbolics.scalarize(norm(ssys.veh.u)))
    zvm = Makie.lines!(b9, retimer.(powered_vel.t), getindex.(u_powered.u, 1))

    b10 = Makie.Axis(f[2, 5], title="Thrust accel. (Body, m/s^2)")
    th_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.th/(ssys.veh.mp))
    zvm = Makie.lines!(b10, retimer.(powered_vel.t), getindex.(th_powered.u, 1), color=:red)
    xvm = Makie.lines!(b10, retimer.(powered_vel.t), getindex.(th_powered.u, 2), color=:green)
    yvm = Makie.lines!(b10, retimer.(powered_vel.t), getindex.(th_powered.u, 3), color=:blue)

    b11 = Makie.Axis(f[3, 5], title="Fuel Mass (kg)", limits=(nothing,(0.0,12000)), xlabel="Time (s)")
    m_powered = sol_res(LinRange(0.5,1.0,100), idxs=ssys.veh.mp - ssys.veh.mdry)
    zvm = Makie.lines!(b11, retimer.(powered_vel.t), m_powered.u)
    f
end