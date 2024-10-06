using Makie, GLMakie
using CSV, DataFrames
using Rotations
using Polyhedra, CDDLib
using StaticArrays, LazyArrays
using LinearAlgebra, LsqFit
using StatsBase
import QHull

data = CSV.read("data/fin_vector_3.csv", DataFrame);


aoa1 = -0.0
aoa2 = 0.0
function points_at(aoa1, aoa2)
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(3.0))
    forces = collect.(eachrow(select(actuation_point, :fx, :fy, :fz)))
    minfrc, min_drag_idx = findmin(el->el[2], ((RotXZ(deg2rad(-aoa1), deg2rad(-aoa2)), ) .* forces))
    @show minfrc
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    return collect.((RotXZ(deg2rad(-aoa1), deg2rad(-aoa2)), ) .* (forces .- (reference_point,)))
end

function points_at(mach, aoa1, aoa2)
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(mach))
        return points_at(actuation_point)
end

function points_at(actuation_point)
    aoa1, aoa2, mach = actuation_point[1, [:aoa1, :aoa2, :mach]]
    forces = collect.(eachrow(select(actuation_point, :fx, :fy, :fz)))
    minfrc, min_drag_idx = findmin(el->el[2], ((RotXZ(deg2rad(-aoa1), deg2rad(-aoa2)), ) .* forces))
    @show mach aoa1 aoa2 minfrc
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    return SVector{3}.(collect.((RotXZ(deg2rad(-aoa1), deg2rad(-aoa2)), ) .* (forces .- (reference_point,)))), actuation_point[min_drag_idx,:]
end

function torques_at(aoa1, aoa2)
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(0.5))
    forces = collect.(eachrow(select(actuation_point, :tx, :ty, :tz)))
    minfrc, min_drag_idx = findmin(el->el[2], ((RotXZ(deg2rad(-aoa1), deg2rad(-aoa2)), ) .* forces))
    @show minfrc
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    return collect.((forces ))
end



scatter(Point3.(torques_at(25.0,0.0)))
scatter(Point3.(points_at(25.0,0.0)))
scatter!(Point3.(points_at(0.0,0.0)))

scatter!(Point3.(points_at(0.0,15.0)))
scatter!(Point3.(points_at(0.0,-10.0) .- ([150, 0, 0], )))


scatter!(Point3.(points_at(0.0,15.0)))
scatter!(Point3.(points_at(0.0,20.0) .+ ([150, 0, 0], )))
scatter!(Point3.(points_at(0.0,25.0)))

aoa_combos = collect.(eachrow(unique(select(data, :mach, :aoa1, :aoa2))))

_isapprox(x::Union{T, AbstractArray{T}}, y::Union{T, AbstractArray{T}}) where {T<:Union{Integer, Rational}} = x == y
_isapprox(x, y) = (Polyhedra.isapproxzero(x) ? Polyhedra.isapproxzero(y) : (Polyhedra.isapproxzero(y) ? Polyhedra.isapproxzero(x) : Polyhedra.isapproxzero(norm(@~ x .- y))))
_isapprox(sx, x::AbstractVector, sy, y::AbstractVector) = 
    (Polyhedra.isapproxzero(x, sx) ? Polyhedra.isapproxzero(y,sy) : (Polyhedra.isapproxzero(y, sy) ? Polyhedra.isapproxzero(x,sx) : Polyhedra.isapproxzero(_scaled_fastnorm(sx, x, sy, y))))
#_scaled_fastnorm(sx, x, sy, y) = sqrt(sum(@~ (sx .* x .- sy .* y) .^ 2))
function _scaled_fastnorm(sx, x, sy, y) # where T 
    acc = zero(typeof(sy))
    @assert length(x) == length(y)
    for i in eachindex(x)
        @inbounds acc += (sx * x[i] - sy * y[i])^2
    end
    return sqrt(acc)
end

function _scalehp(h1, h2)
    s1 = sum(@~ abs.(h1.a)) + abs(h1.β)
    s2 = sum(@~ abs.(h2.a)) + abs(h2.β)
    s2, s1 # (h1.a*s2, h1.β*s2), (h2.a*s1, h2.β*s1) # SCALE FOR 1, SCALE FOR 2
end
function _scalehph(h1, h2)
    s1 = sum(@~ abs.(h1.a)) + abs(h1.β)
    s2 = sum(@~ abs.(h2.a)) + abs(h2.β)
    (h1.a*s2, h1.β*s2), (h2.a*s1, h2.β*s1) # SCALE FOR 1, SCALE FOR 2
end
function Base.:(==)(h1::HyperPlane, h2::HyperPlane)
    s1,s2 = _scalehp(h1, h2)
    (all(@~ s1 .* h1.a .== s2 .* h2.a) && s1 * h1.β == s2 * h2.β) || (all(@~ s1 * h1.a .== s2 * -h2.a) && s1 * h1.β == s2 * -h2.β)
end
function Base.isapprox(h1::HyperPlane, h2::HyperPlane)
    s1,s2 = _scalehp(h1, h2)
    (_isapprox(s1, h1.a, s2, h2.a) && _isapprox(s1, h1.β, s2, h2.β)) || 
    (_isapprox(s1, h1.a, -s2, h2.a) && _isapprox(s1, h1.β, -s2, h2.β))
end
function Base.isapprox(h1::HalfSpace, h2::HalfSpace)
    s1,s2 = _scalehp(h1, h2)
    _isapprox(s1, h1.a, s2, h2.a) && _isapprox(s1, h1.β, s2, h2.β)
end

function Polyhedra.isapproxzero(x::AbstractVector{T}; kws...) where {T<:Real} 
    lmax = zero(T)
    for el in x
        v = abs(el)
        if v > lmax
            lmax = v
        end
    end
    return Polyhedra.isapproxzero(lmax; kws...)
end

function Polyhedra.isapproxzero(x::AbstractVector{T}, scale::T; kws...) where {T<:Real} 
    lmax = zero(T)
    for el in x
        v = abs(el*scale)
        if v > lmax
            lmax = v
        end
    end
    return Polyhedra.isapproxzero(lmax; kws...)
end
function extract_aero_data()
    rdf = DataFrame(
        :aoa1 => [],
        :aoa2 => [],
        :mach => [],
        :cla1 => [],
        :cl_upper1 => [],
        :cl_lower1 => [],
        :cla2 => [],
        :cl_upper2 => [],
        :cl_lower2 => [],
        :bfx_b => [],
        :bfy_b => [],
        :bfz_b => [],
        :btx_b => [],
        :bty_b => [],
        :btz_b => [],
    )
    for actuation_point in groupby(data, [:aoa1, :aoa2, :mach])
        pts, rp = points_at(actuation_point)
        ph = polyhedron(vrep(pts), QHull.Library())
        removevredundancy!(ph)
        ph1 = eliminate(ph, [1], ProjectGenerators())
        removevredundancy!(ph1)
        ph2 = eliminate(ph, [3], ProjectGenerators())
        removevredundancy!(ph2)


        function get_fit_points(polyhedra, dir)
            fit_points = []
            for pt_ind in eachindex(points(polyhedra))
                if any(map(sp -> dot(sp.a,dir), incidenthalfspaces(polyhedra, pt_ind)) .> 0.25)
                    push!(fit_points, get(polyhedra, pt_ind))
                end
            end
            return fit_points
        end

        ph1_xform = sort(map(v->[0 1; 1 0] * v, get_fit_points(ph1, [-1.0, 0.0])), by=first)
        ph2_pts = sort(get_fit_points(ph2, [0.0, -1.0]), by=first)

        function quad_fit(pts)
            ires = LsqFit.curve_fit((t,p)->p[1] * t.^2, first.(pts), last.(pts), [0.01])
            pos_lmax = maximum(last.(filter(p->p[1] > 0, pts)))
            neg_lmax = maximum(last.(filter(p->p[1] < 0, pts)))
            return (coef=ires.param[1], lmin=neg_lmax, lmax=pos_lmax)
        end
        r1 = quad_fit(ph1_xform)
        r2 = quad_fit(ph2_pts)
        
        if false # (actuation_point[1, :mach] > 1.0)
            f = Figure()
            ax = Axis(f[1,1])
            ax2 = Axis(f[2,1])
            Makie.mesh!(ax, Polyhedra.Mesh(ph1), color=:blue)
            Makie.mesh!(ax2, Polyhedra.Mesh(ph2), color=:red)
            scatter!(ax,Point2.(get_fit_points(ph1, [-1.0, 0.0])))
            scatter!(ax2,Point2.(get_fit_points(ph2, [0.0, -1.0])))
            lines!(ax, ((l)->r1.coef*l^2).(first.(ph1_xform)),first.(ph1_xform))
            lines!(ax2, first.(ph2_pts), ((l)->r2.coef*l^2).(first.(ph2_pts)))
            display(f)
            return 
        end
        push!(rdf, (
            aoa1=actuation_point[1, :aoa1],
            aoa2=actuation_point[1, :aoa2],
            mach=actuation_point[1, :mach],
            cla1=r1.coef,
            cl_upper1=r1.lmax,
            cl_lower1=r1.lmin,
            cla2=r2.coef,
            cl_upper2=r2.lmax,
            cl_lower2=r2.lmin,
            bfx_b=rp[:fx],
            bfy_b=rp[:fy],
            bfz_b=rp[:fz],
            btx_b=rp[:tx],
            bty_b=rp[:ty],
            btz_b=rp[:tz],
        ))
        if length(res) > 10
            #break 
        end
    end
    return rdf
end

function no_stall_approximation(machdf, liftmap)
    limits = DataFrame([
        :aoa1 => [],
        :aoa2 => [],
        :nc1 => [], # mapping from [-1,1] to lift space
        :upper_lim1 => [], # no-stall limits on [-1,1] lift allocation (before mapping with nc1)
        :lower_lim1 => [],
        :nc2 => [],
        :upper_lim2 => [],
        :lower_lim2 => []
    ])
    for rd in eachrow(machdf)
        #compute the critical points for one plane (determined by af/cf/upper_col/lower_col) in 
        # normalized lift space based on the opposing aoa (in aoa_dep)
        function compute_critical_points(
            af, cf, aoa_dep, upper_col, lower_col)
            qc = liftmap[af] * cosd(rd[aoa_dep]) + liftmap[cf]
            lift_upper = sqrt(rd[upper_col] ./ qc)
            lift_lower = sqrt(rd[lower_col] ./ qc)
            normalization_constant = max(lift_upper, lift_lower)
            upper_lim = lift_upper/normalization_constant
            lower_lim = lift_lower/normalization_constant
            return (nc=normalization_constant, ul=upper_lim, ll=lower_lim)
        end
        plane1 = compute_critical_points(:cl1_lin, :cl1_const, :aoa2, :cl_upper1, :cl_lower1)
        plane2 = compute_critical_points(:cl2_lin, :cl2_const, :aoa1, :cl_upper2, :cl_lower2)
        push!(limits, (aoa1=rd[:aoa1], aoa2=rd[:aoa2], # upper/lower are really misleading names, should change them. really positive lmax and negative lmax.
            nc1=plane1.nc, upper_lim1=plane1.ul, lower_lim1=plane1.ll,
            nc2=plane2.nc, upper_lim2=plane2.ul, lower_lim2=plane2.ll))
    end

    function limits_for(aoa_component, col, side; show_polytope=false, axis=nothing)
        ph = polyhedron(vrep(collect.(eachrow(hcat(limits[:,aoa_component], limits[:, col])))), QHull.Library())
        removevredundancy!(ph)
        function get_control_limits(polyhedra, side)
            pts = collect(points(polyhedra))
            # first check to see if the entire space is allocatable (e.g. at 1.0 all the way across)
            lmin = minimum(pt -> pt[2] > 0.99 ? pt[1] : Inf, pts)
            lmax = maximum(pt -> pt[2] > 0.99 ? pt[1] : -Inf, pts)
            if (lmin < -24.95 && lmax > 24.95)
                return nothing
            end
            # conservatively identify the point in the aoa_deg vs normalized lift plot where it starts changing
            _, kneei= findmax(pt -> pt[2] > 0.99 ? side*pt[1] : -Inf, pts)
            kneept = pts[kneei]
            _, endi = findmin(pt -> pt[2], pts)
            endpt = pts[endi]
            dx,dy = kneept - endpt
            dir = normalize([side*dy, -side*dx])
            offs = dot(dir, kneept)
            return dir, offs
        end
        res = get_control_limits(ph, side)
        if show_polytope
            Makie.mesh!(axis, Polyhedra.Mesh(ph), color=:blue)
            if !isnothing(res)
                calced = polyhedron(HalfSpace(res[1], res[2]) ∩ HalfSpace([-1, 0], 25) ∩ HalfSpace([1, 0], 25) ∩ HalfSpace([0, 1], 1) ∩ HalfSpace([0, -1], 0) )
            else
                calced = polyhedron(HalfSpace([-1, 0], 25) ∩ HalfSpace([1, 0], 25) ∩ HalfSpace([0, 1], 1) ∩ HalfSpace([0, -1], 0) )
            end
            Makie.mesh!(axis, Polyhedra.Mesh(calced), color=(:red, 0.5))
        end
        return !isnothing(res) ? res : ([0.0,0.0],NaN)
    end
    #WLOG lower and upper should be symmetric and aoa2 vs aoa1 should be literally the same
    #there's some signedness issue where plane 2 comes out symmetric to plane 1 for some reason
    #f=Figure()
    #ax1 = Axis(f[1,1])
    # the side-edness needs to be changed for each vehicle
    ul1 = limits_for(:aoa1, :upper_lim1, -1.0)#; show_polytope=true, axis=ax1)
    #ax2 = Axis(f[1,2])
    ll1 = limits_for(:aoa1, :lower_lim1, 1.0)#; show_polytope=true, axis=ax2)
    #ax3 = Axis(f[2,2])
    ul2 = limits_for(:aoa2, :upper_lim2, 1.0)#; show_polytope=true, axis=ax3)
    #ax4 = Axis(f[2,1])
    ll2 = limits_for(:aoa2, :lower_lim2, -1.0)#; show_polytope=true, axis=ax4)
    #display(f)
    dela = ul1 .- ll2 
    delb = ul2 .- ll1
    foma = sum(abs.(dela[1])) + abs(dela[2])
    fomb = sum(abs.(delb[1])) + abs(delb[2])
    if (foma > 1e-3) || (fomb > 1e-3)
        @warn "Symmetry assumption violated foma: $foma fomb: $fomb mach: $(first(machdf)[:mach])"
        @show ul1 ll1 ul2 ll2
    end
    return (ul1=ul1, ll1=ll1, ul2=ul2, ll2=ll2)
end

function control_approxmation(aero_data)
    aero_control_data = DataFrame([
        :mach => [],
        :cl1_lin => [],
        :cl1_const => [],
        :cl2_lin => [],
        :cl2_const => [],
        :plmax1_dir_aoa => [],
        :plmax1_dir_ndl => [],
        :plmax1_offs => [],
        :nlmax1_dir_aoa => [],
        :nlmax1_dir_ndl => [],
        :nlmax1_offs => [],
        :plmax2_dir_aoa => [],
        :plmax2_dir_ndl => [],
        :plmax2_offs => [],
        :nlmax2_dir_aoa => [],
        :nlmax2_dir_ndl => [],
        :nlmax2_offs => []]
    )
    liftmaps=[]
    lims=DataFrame([:ul1=>[],:ll1=>[],:ul2=>[],:ll2=>[]])
    for machg in groupby(aero_data, :mach)
        f1 = LsqFit.curve_fit((t,p)->p[1] * t .+ p[2], cosd.(machg[:,:aoa2]), machg[:,:cla1], [0.01,0.0]).param
        f2 = LsqFit.curve_fit((t,p)->p[1] * t .+ p[2], cosd.(machg[:,:aoa1]), machg[:,:cla2], [0.01,0.0]).param
        liftmap = (mach=machg[1, :mach], 
            cl1_lin=f1[1], cl1_const=f1[2],
            cl2_lin=f2[1], cl2_const=f2[2])
        push!(lims, no_stall_approximation(machg, liftmap))
        push!(liftmaps, liftmap)
    end
    for (rowno, limits) in enumerate(eachrow(lims))
        isgood(pr) = !isnan(pr[2])
        function findgoodpair(field)
            lastgood=nothing
            for testrow=rowno-1:-1:1
                if isgood(lims[testrow,field])
                    lastgood = lims[testrow,field]
                    break
                end
            end
            nextgood=nothing
            for testrow=rowno+1:nrow(lims)
                if isgood(lims[testrow,field])
                    nextgood = lims[testrow,field]
                    break
                end
            end
            return lastgood, nextgood
        end
        function fixup(ex, col, corner)
            if isgood(ex)
                return ex
            end
            lastgood, nextgood = findgoodpair(col)
            if isnothing(lastgood) || isnothing(nextgood)
                throw("cannot find no stall polytopes to interpolate between")
            end
            dir = (lastgood[1] + nextgood[1])./2
            dir = dir / norm(dir)
            mag = dot(corner, dir)
            return (dir, mag)
        end
        # new vehicle: change the vertex positions here 
        ul1 = fixup(limits[:ul1], :ul1, [-25.0, 1.0])
        ll1 = fixup(limits[:ll1], :ll1, [25.0, 1.0])
        ul2 = fixup(limits[:ul2], :ul2, [25.0, 1.0])
        ll2 = fixup(limits[:ll2], :ll2, [-25.0, 1.0])
        push!(aero_control_data, (
            plmax1_dir_aoa = ul1[1][1],
            plmax1_dir_ndl = ul1[1][2],
            plmax1_offs = ul1[2],
            nlmax1_dir_aoa = ll1[1][1],
            nlmax1_dir_ndl = ll1[1][2],
            nlmax1_offs = ll1[2],
            plmax2_dir_aoa = ul2[1][1],
            plmax2_dir_ndl = ul2[1][2],
            plmax2_offs = ul2[2],
            nlmax2_dir_aoa = ll2[1][1],
            nlmax2_dir_ndl = ll2[1][2],
            nlmax2_offs = ll2[2],
            liftmaps[rowno]...
        ))
    end
    return aero_control_data
end

aero_data = extract_aero_data()
# lift polars are linnear functions of the cos(AoA) that the fin is tilted at by the other axis...
control_data = control_approxmation(aero_data)

body_aero = DataFrame(
    :mach => [],
    :aoa1 => [],
    :aoa2 => [],
    :aoa_net => [],
    :lift => [],
    :drag => [],
    :frc_err => []
)
for expt in eachrow(aero_data)
    aoa1, aoa2, mach = expt[[:aoa1, :aoa2, :mach]]
    Rbi = RotXZ(deg2rad(-aoa1), deg2rad(-aoa2))
    frc_windframe = Rbi * collect(expt[[:bfx_b, :bfy_b, :bfz_b]])
    bodydir = Rbi * [0,1.0,0]
    up = [0.0,1.0,0.0]
    bv1=cross(up, bodydir)
    bv1=bv1/norm(bv1)
    bv2=cross(up,bv1/norm(bv1))
    drag = dot(up, frc_windframe)
    frc_err = dot(bv1, frc_windframe)
    lift = dot(bv2, frc_windframe)
    aa_rbi = AngleAxis(Rbi)
    push!(body_aero, (mach=mach,aoa1=aoa1,aoa2=aoa2,aoa_net=rad2deg(aa_rbi.theta), lift=isnan(lift) ? 0.0 : lift, drag=isnan(drag) ? 0.0 : drag,frc_err=frc_err/drag))
end
body_aero_out = combine(groupby(select(body_aero, Not([:aoa1, :aoa2])), [:mach, :aoa_net]), :lift => mean => :lift, :drag => mean => :drag)

scatter((df -> Point2.(df[:, :aoa_net], df[:, :lift]))(body_aero_out |> filter(:mach => ==(2.5))))
scatter((df -> Point2.(df[:, :mach], df[:, :lift]))(body_aero_out |> filter(:aoa_net => ==(5.0))))
