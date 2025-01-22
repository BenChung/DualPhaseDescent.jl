using Makie, GLMakie
using CSV, Parquet2, DataFrames
using Rotations
using Polyhedra, CDDLib
using StaticArrays, LazyArrays
using LinearAlgebra, LsqFit, GLM
using StatsBase
using DelaunayTriangulation, ScatteredInterpolation
import QHull
import Interpolations

#data = CSV.read("data/fin_vector_3.csv", DataFrame);
data = DataFrame(Parquet2.Dataset("data/fin_vector_6.pq2"); copycols=false)

function lookat_rmat(f, tgt) # moller and hughes, actually works???
    v = cross(f, tgt)
    c = dot(f, tgt)
    h = 1/(1+c)
    return I(3) * c + h * v * v' + [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end


planarize(v) = [v[1], 0.0, v[3]]/norm([v[1], 0.0, v[3]])
rebasis(b,v) = [dot(v,b[1]),dot(v,b[2]),dot(v,b[3])]
#scatter(Point3.(rebasis.(([[1.0,0.0,0.0], [0.0,1.0,0.0], planarize(RotZX(deg2rad(-20.0), deg2rad(-25.0)) * [0.0,0.0,1.0])], ), points_at(0.9 ,20.0, -25.0))))

aoa1 = -0.0
aoa2 = 0.0
function points_at(aoa1, aoa2)
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(3.0))
    forces = collect.(eachrow(select(actuation_point, :fx, :fy, :fz)))
    derot_mat = lookat_rmat(normalize(collect(actuation_point[1, [:vx, :vy, :vz]])), [0,-1,0])
    minfrc, min_drag_idx = findmin(el->el[2], ((derot_mat, ) .* forces))
    @show minfrc
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    return collect.((derot_mat, ) .* (forces .- (reference_point,)))
end
function points_at(mach, aoa1, aoa2)
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(mach))
    forces = collect.(eachrow(select(actuation_point, :fx, :fy, :fz)))
    derot_mat = lookat_rmat(normalize(collect(actuation_point[1, [:vx, :vy, :vz]])), [0,-1,0])
    minfrc, min_drag_idx = findmin(el->el[2], ((derot_mat, ) .* forces))
    @show minfrc
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    @show derot_mat * reference_point
    pts = collect.((derot_mat, ) .* (forces .- (reference_point,)))
    #deskew = rebasis.(([[1.0,0.0,0.0], [0.0,1.0,0.0], planarize(RotZX(deg2rad(aoa1), deg2rad(-aoa2)) * [0.0,0.0,1.0])], ), pts)
    return pts
end

function forces_at(mach, aoa1, aoa2, thresh)
    pt = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(mach))|>
            filter(:acty => x -> abs.(x.-thresh) .< 0.2)
    derot_mat = lookat_rmat(normalize(collect(pt[1, [:vx, :vy, :vz]])), [0,-1,0])
    forces = collect.(eachrow(select(pt, :fx, :fy, :fz)))
    return (derot_mat, ) .* forces
end


function forces_at(mach, aoa1, aoa2)
    pt = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(mach))|>
            filter(:actx => ==(0.0)) |> 
            filter(:actz => ==(0.0))
    forces = collect.(eachrow(select(pt, :fx, :fy, :fz)))
    return (derot_mat, ) .* forces
end

function points_at(actuation_point)
    derot_mat = lookat_rmat(normalize(collect(actuation_point[1, [:vx, :vy, :vz]])), [0,-1,0])
    aoa1, aoa2, mach = actuation_point[1, [:aoa1, :aoa2, :mach]]
    forces = collect.(eachrow(select(actuation_point, :fx, :fy, :fz)))
    acts = collect.(eachrow(select(actuation_point, :actx, :actz)))
    minfrc, min_drag_idx = findmin(el->el[2], ((derot_mat, ) .* forces))
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    pts = collect.((derot_mat, ) .* (forces .- (reference_point,)))
    deskew = rebasis.(([[1.0,0.0,0.0], [0.0,1.0,0.0], planarize(RotZX(deg2rad(aoa1), deg2rad(-aoa2)) * [0.0,0.0,1.0])], ), pts)
    return SVector{3}.(deskew), actuation_point[min_drag_idx,:]
end

function torques_at(aoa1, aoa2)
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(0.5))
    derot_mat = lookat_rmat(normalize(collect(actuation_point[1, [:vx, :vy, :vz]])), [0,-1,0])
    forces = collect.(eachrow(select(actuation_point, :tx, :ty, :tz)))
    minfrc, min_drag_idx = findmin(el->el[2], ((derot_mat, ) .* forces))
    @show minfrc
    reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
    return collect.((forces ))
end

aoa_combos = collect.(eachrow(unique(select(data, :mach, :aoa1, :aoa2))))

_isapproxzero(sx, x) = sx * x ≈ 0.0
_isapproxzero(sx, x::Vector) = all(@~ _isapproxzero.((sx, ), x))
_isapproxzero(x) = x ≈ 0.0
_isapprox(x::Union{T, AbstractArray{T}}, y::Union{T, AbstractArray{T}}) where {T<:Union{Integer, Rational}} = x == y
_isapprox(x, y) = (Polyhedra.isapproxzero(x) ? Polyhedra.isapproxzero(y) : (Polyhedra.isapproxzero(y) ? Polyhedra.isapproxzero(x) : Polyhedra.isapproxzero(norm(@~ x .- y))))
_isapprox(sx, x::AbstractVector, sy, y::AbstractVector) = 
    (_isapproxzero(sx, x) ? _isapproxzero(sy, y) : (_isapproxzero(sy,y) ? _isapproxzero(sx, x) : _isapproxzero(_scaled_fastnorm(sx, x, sy, y))))
_isapprox(sx, x::Float64, sy, y::Float64) = 
        (_isapproxzero(x, sx) ? _isapproxzero(y,sy) : (_isapproxzero(y, sy) ? _isapproxzero(x,sx) : _isapproxzero(abs(sx * x - sy * y))))
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
        :ref_vel => [],
    )
    for actuation_point in groupby(data, [:aoa1, :aoa2, :mach])
        #if actuation_point[1, :mach] != 1.2 || actuation_point[1, :aoa2] != 25.0 || actuation_point[1, :aoa1] != 0.0
            #continue 
        #end
        pts, rp = points_at(actuation_point)

        ipts = pts
        qh1 = convex_hull(map(p->(p[2], p[3]), pts))
        qh2 = convex_hull(map(p->(p[1], p[2]), pts))

        @show actuation_point[1, [:aoa1, :aoa2, :mach]]


        function get_fit_points2(qh, dir)
            pts = DelaunayTriangulation.get_points(qh)[DelaunayTriangulation.get_vertices(qh)][1:end-1]
            npts = length(pts)
            fit_points = SVector{2, Float64}[]
            for pt_ind in eachindex(pts)
                prev_ind = (pt_ind - 1 + npts)%npts
                next_ind = (pt_ind + 1 + npts)%npts
                if prev_ind == 0 prev_ind = npts end
                if next_ind == 0 next_ind = npts end
                normal1 = RotMatrix(π/2)*SVector(pts[prev_ind] .- pts[pt_ind])
                normal1 = normal1/norm(normal1)
                normal2 = RotMatrix(-π/2)*SVector(pts[next_ind] .- pts[pt_ind])
                normal2 = normal2/norm(normal2)
                threshold = 0.8
                if dot(dir, normal1) > threshold || dot(dir, normal2) > threshold
                    push!(fit_points, SVector{2, Float64}(pts[pt_ind]))
                end
            end
            return fit_points
        end

        ph1_xform = sort(map(v->[0 1; 1 0] * v, get_fit_points2(qh1, [-1.0, 0.0])), by=first)
        ph2_pts = sort(get_fit_points2(qh2, [0.0, -1.0]), by=first)

        function quad_fit(pts)
            ires = LsqFit.curve_fit((t,p)->p[1] * t.^2, first.(pts), last.(pts), [0.01])
            pos_lmax = maximum(first.(filter(p->p[1] > 0, pts)))
            neg_lmax = minimum(first.(filter(p->p[1] < 0, pts)))
            return (coef=ires.param[1], lmin=neg_lmax, lmax=pos_lmax)
        end
        r1 = quad_fit(ph1_xform)
        r2 = quad_fit(ph2_pts)
        
        if false #actuation_point[1, :mach] == 1.2 && actuation_point[1, :aoa2] == 25.0 && actuation_point[1, :aoa1] == 0.0
            f = Figure()
            ax = Axis(f[1,1])
            ax2 = Axis(f[2,1])
            #Makie.mesh!(ax, Polyhedra.Mesh(ph1), color=:blue)
            #Makie.mesh!(ax2, Polyhedra.Mesh(ph2), color=:red)
            scatter!(ax,map(p->(p[2], p[3]), ipts))
            scatter!(ax2,map(p->(p[1], p[2]), ipts))
            scatter!(ax,(get_fit_points2(qh1, [-1.0, 0.0])))
            scatter!(ax2,(get_fit_points2(qh2, [0.0, -1.0])))
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
            ref_vel=collect(rp[[:vx, :vy, :vz]])
        ))
    end
    return rdf
end

function control_limits(limits_df, aero_data, liftmap)
    nc1 = -Inf
    nc2 = -Inf
    for rd in eachrow(aero_data)
        function compute_normalization(
            af, cf, aoa_dep, upper_col, lower_col)
            return max(abs(rd[upper_col]), abs(rd[lower_col]))
        end
        nc1 = max(nc1, compute_normalization(:cl1_lin, :cl1_const, :aoa2, :cl_upper1, :cl_lower1))
        nc2 = max(nc2, compute_normalization(:cl2_lin, :cl2_const, :aoa1, :cl_upper2, :cl_lower2))
    end
    for rd in eachrow(aero_data)
        #compute the critical points for one plane (determined by af/cf/upper_col/lower_col) in 
        # normalized lift space based on the opposing aoa (in aoa_dep)
        function compute_critical_points(
            af, cf, aoa_dep, upper_col, lower_col, nc)
            upper_lim = rd[upper_col]/nc
            lower_lim = -rd[lower_col]/nc
            return (nc=nc, ul=upper_lim, ll=lower_lim)
        end
        plane1 = compute_critical_points(:cl1_lin, :cl1_const, :aoa2, :cl_upper1, :cl_lower1, nc1)
        plane2 = compute_critical_points(:cl2_lin, :cl2_const, :aoa1, :cl_upper2, :cl_lower2, nc2)
        push!(limits_df, (mach=rd[:mach], aoa1=rd[:aoa1], aoa2=rd[:aoa2], # upper/lower are really misleading names, should change them. really positive lmax and negative lmax.
            upper_lim1=plane1.ul, lower_lim1=plane1.ll,
            upper_lim2=plane2.ul, lower_lim2=plane2.ll))
    end
    return (nc1=nc1, nc2=nc2)
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
    #display(scatter(limits[:, :aoa2], limits[:, :nc1]))
    #throw("bt")
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
        :cl1_scale => [], # mapping from [-1,1] to lift space
        :cl2_lin => [],
        :cl2_const => [],
        :cl2_scale => []]
    )
    limits = DataFrame([
        :mach => [],
        :aoa1 => [],
        :aoa2 => [],
        :upper_lim1 => [], # no-stall limits on [-1,1] lift allocation (before mapping with nc1)
        :lower_lim1 => [],
        :upper_lim2 => [],
        :lower_lim2 => []
    ])
    lims=DataFrame([:ul1=>[],:ll1=>[],:ul2=>[],:ll2=>[]])
    for machg in groupby(aero_data, :mach)
        f1 = LsqFit.curve_fit((t,p)->p[1] * t .+ p[2], cosd.(machg[:,:aoa2]), machg[:,:cla1], [0.01,0.0]).param
        f2 = LsqFit.curve_fit((t,p)->p[1] * t .+ p[2], cosd.(machg[:,:aoa1]), machg[:,:cla2], [0.01,0.0]).param
        liftmap = (mach=machg[1, :mach], 
            cl1_lin=f1[1], cl1_const=f1[2],
            cl2_lin=f2[1], cl2_const=f2[2])
        nc1,nc2 = control_limits(limits, machg, liftmap)
        rvelsq = mean(norm.(machg[!, :ref_vel]))^2
        #push!(lims, no_stall_approximation(machg, liftmap))
        push!(aero_control_data, (liftmap..., cl1_scale=nc1/rvelsq, cl2_scale=nc2/rvelsq))
    end
    return aero_control_data, limits
end

function make_body_aero(aero_data)
    body_aero = DataFrame(
        :mach => [],
        :aoa1 => [],
        :aoa2 => [],
        :aoa_net => [],
        :refvel => [],
        :lift_raw => [],
        :drag_raw => [],
        :trq_raw => [],
        :lift => [], # lift: normalized by csc(alpha)/v^2. in the direction of the body vector from the velocity vector (? checkme)
        :drag => [], # drag: normalized by 1/v^2. opposite the direction of the velocity vector
        :trq => [], # torque: normalized by csc(alpha)/v^2. in the direction of the body cross velocity vector/ (? checkme signs)
        :frc_err => []
    )
    for expt in eachrow(aero_data)
        aoa1, aoa2, mach = expt[[:aoa1, :aoa2, :mach]]
        derot_mat = lookat_rmat(normalize(collect(expt[:ref_vel])), [0,-1,0])
        Rbi = derot_mat
        frc_windframe = Rbi * collect(expt[[:bfx_b, :bfy_b, :bfz_b]])
        trq_windframe = Rbi * collect(expt[[:btx_b, :bty_b, :btz_b]])
        bodydir = Rbi * [0, 1.0, 0] 
        up = [0.0, 1.0, 0.0]
        bv1 = cross(up, bodydir)
        bv1 = bv1 / norm(bv1)
        bv2 = cross(up, bv1 / norm(bv1))
        aa_rbi = AngleAxis(Rbi)
        drag = dot(up, frc_windframe) / norm(expt[:ref_vel])^2
        frc_err = dot(bv1, frc_windframe)
        lift = csc(aa_rbi.theta) * dot(bv2, frc_windframe) / norm(expt[:ref_vel])^2
        trq = csc(aa_rbi.theta) * dot(bv1, trq_windframe) / norm(expt[:ref_vel])^2
        #trq_err = norm(trq_windframe - trq*bv1)/trq
        push!(body_aero, (mach=mach, aoa1=aoa1, aoa2=aoa2, aoa_net=round(rad2deg(aa_rbi.theta); digits=8), refvel = norm(expt[:ref_vel]),
        lift_raw=dot(bv2, frc_windframe), drag_raw = dot(up, frc_windframe), trq_raw = dot(bv1, trq_windframe),
        lift=isnan(lift) ? 0.0 : lift, drag=isnan(drag) ? 0.0 : drag, frc_err=frc_err / drag, trq=isnan(trq) ? 0.0 : trq))
    end
    return combine(groupby(select(body_aero, Not([:aoa1, :aoa2])), [:mach, :aoa_net]), 
    :lift => mean => :lift, :drag => mean => :drag, :trq => mean => :trq, 
    :lift_raw => mean => :lift_raw, :drag_raw => mean => :drag_raw, :trq_raw => mean => :trq_raw,
    :refvel => mean => :refvel)
end

function build_inverse_control_map()
    actuation_points = groupby(data, [:aoa1, :aoa2, :mach])
    cpt = 1
    res = DataFrame(
        :mach => [],
        :aoa1 => [],
        :aoa2 => [],
        :axis => [],
        :real_demand => [],
        :demand => [],
        :control => []
    )
    for actuation_point in keys(actuation_points)
        (aoa1, aoa2, mach) = actuation_point
        if mach < 0.5 || aoa1 != 5 || aoa2 != 25 || mach != 1.2 
            #continue 
        end
        rel_pts = filter(:acty => ==(0.0), actuation_points[actuation_point])
        approx = control_eff |> filter(:mach => ==(mach)) |> first
        rv = mean((body_data |> filter(:mach => ==(mach)))[:,:refvel])
        cc1 = approx[:cl1_lin] * cosd(aoa2) + approx[:cl1_const]
        cc2 = approx[:cl2_lin] * cosd(aoa1) + approx[:cl2_const]
        lims = control_lims |> filter(:mach => ==(mach)) |> filter(:aoa1 => ==(aoa1)) |> filter(:aoa2 => ==(aoa2)) |> first
        derot_mat = lookat_rmat(normalize(collect(rel_pts[1, [:vx, :vy, :vz]])), [0,-1,0])

        forces = collect.(eachrow(select(rel_pts, :fx, :fy, :fz)))
        minfrc, min_drag_idx = findmin(el->el[2], ((derot_mat, ) .* forces))
        reference_point = (rel_pts[min_drag_idx,[:fx,:fy,:fz]]) |> collect
        r1 = derot_mat
        r2 = RotXY(deg2rad(90), deg2rad(90))
        adjusted = transform(rel_pts, [:fx,:fy,:fz] => ByRow((f...) -> r2 * r1 * ([f..., ] - reference_point)) => :ftf)
        # lift = rv^2 * cl_scale * lv
        scales = collect(approx[[:cl1_scale, :cl2_scale]])
        lower_lim = (-).(collect(lims[[:lower_lim1, :lower_lim2]]))
        upper_lim = (collect(lims[[:upper_lim1, :upper_lim2]]))
        drag_scales = [cc1, cc2]

        adjusted = transform(adjusted, :ftf => ByRow(frc -> frc[[1,2]] ./ (rv^2 .* scales)) => :fcmd)
        adjusted = transform(adjusted, :fcmd => ByRow(cmd -> cmd[1]) => :fcmd_1)
        adjusted = transform(adjusted, :fcmd => ByRow(cmd -> cmd[2]) => :fcmd_2)
        #adjusted = filter(:fcmd => (cmd -> all(lower_lim .<= cmd) && all(cmd .<= upper_lim)), adjusted)
        adjusted = transform(adjusted, :fcmd => ByRow(cmd -> sum(drag_scales .* (rv^2 .* scales .* cmd) .^2)) => :induced)
        adjusted = transform(adjusted, [:induced, :ftf] => ByRow((pdrg, rfrc) -> (pdrg - rfrc[3])) => :drg_err)
        not_insane = filter(:drg_err => (d->abs(d) < 10.0), adjusted)

        #=
        f = scatter(not_insane[:,:actz], not_insane[:, :fcmd_1])
        scatter!(not_insane[:,:actx], not_insane[:, :fcmd_2])
        display(f)
        return
        =#
        knots = LinRange(-1.0, 1.0, 21)
        cmd2_data = combine(groupby(not_insane, :actx), :fcmd_2 => mean => :fcmd_2)
        sort!(cmd2_data, order(:fcmd_2))
        itp_2 = Interpolations.extrapolate(Interpolations.interpolate(cmd2_data[:, :fcmd_2], cmd2_data[:, :actx], Interpolations.LinearMonotonicInterpolation()), Interpolations.Linear())
        push!(cmd2_data, [itp_2(lower_lim[2]), lower_lim[2]])
        push!(cmd2_data, [itp_2(upper_lim[2]), upper_lim[2]])
        sort!(cmd2_data, order(:fcmd_2))
        itp_2 = Interpolations.extrapolate(Interpolations.interpolate(cmd2_data[:, :fcmd_2], cmd2_data[:, :actx], Interpolations.LinearMonotonicInterpolation()), Interpolations.Linear())
        
        for kt in knots
            un = (kt+1)/2
            spt = (1-un) * lower_lim[2] + un*upper_lim[2]
            push!(res, [
                mach, aoa1, aoa2, :actx, spt, kt, itp_2(spt)
            ])
        end
        
        #display(groupby(not_insane, :actz))
        cmd1_data = combine(groupby(not_insane, :actz), :fcmd_1 => mean => :fcmd_1)
        sort!(cmd1_data, order(:fcmd_1))
        itp_1 = Interpolations.extrapolate(Interpolations.interpolate(cmd1_data[:, :fcmd_1], cmd1_data[:, :actz], Interpolations.LinearMonotonicInterpolation()), Interpolations.Linear())
        push!(cmd1_data, [itp_1(lower_lim[1]), lower_lim[1]])
        push!(cmd1_data, [itp_1(upper_lim[1]), upper_lim[1]])
        sort!(cmd1_data, order(:fcmd_1))
        itp_1 = Interpolations.extrapolate(Interpolations.interpolate(cmd1_data[:, :fcmd_1], cmd1_data[:, :actz], Interpolations.LinearMonotonicInterpolation()), Interpolations.Linear())
        
        for kt in knots
            un = (kt+1)/2
            spt = (1-un) * lower_lim[1] + un*upper_lim[1]
            push!(res, [
                mach, aoa1, aoa2, :actz, spt, kt, itp_1(spt)
            ])
        end

        if cpt % 10 == 0
            @show cpt 
        end
        cpt += 1
    end
    return res
end


(aoa1, aoa2, mach) = (0.0, 0.0, 1.2)
rel_pts = data |> filter(:aoa1 => ==(aoa1)) |> filter(:aoa2 => ==(aoa2)) |> filter(:mach => ==(mach))
approx = control_eff |> filter(:mach => ==(mach)) |> first

ua = 0.8
vel = norm(rel_pts[1, [:vx, :vy, :vz]])
lift_frc = approx[:cl1_scale] * ua * vel^2
cc1 = approx[:cl1_lin] * cosd(aoa2) + approx[:cl1_const]
drg_frc = cc1*lift_frc^2

aero_data = extract_aero_data()
# lift polars are linear functions of the cos(AoA) that the fin is tilted at by the other axis...
control_eff, control_lims = control_approxmation(aero_data)
body_data = make_body_aero(aero_data)
inv_ctrl_map = build_inverse_control_map()

function make_inv_ctrl_lookup(invmap, axis)
    dat = invmap |> filter(:axis => ==(axis))
    mach_kts = unique(dat[:, :mach]); mach_map = Dict(mach_kts .=> eachindex(mach_kts))
    aoa1_kts = unique(dat[:, :aoa1]); aoa1_map = Dict(aoa1_kts .=> eachindex(aoa1_kts))
    aoa2_kts = unique(dat[:, :aoa2]); aoa2_map = Dict(aoa2_kts .=> eachindex(aoa2_kts))
    demand_kts = unique(dat[:, :demand]); demand_map = Dict(demand_kts .=> eachindex(demand_kts))
    lut = zeros((length(mach_kts), length(aoa1_kts), length(aoa2_kts), length(demand_kts)))
    for row in eachrow(dat)
        mind = mach_map[row[:mach]]
        a1ind = aoa1_map[row[:aoa1]]
        a2ind = aoa2_map[row[:aoa2]]
        dind = demand_map[row[:demand]]
        lut[mind, a1ind, a2ind, dind] = row[:control]
    end
    itp = Interpolations.interpolate((mach_kts, aoa1_kts, aoa2_kts, demand_kts), lut, Interpolations.Gridded(Interpolations.Linear()))
    return function (mach, aoa1, aoa2, demand)
        lims = control_lims |> filter(:mach => ==(mach)) |> filter(:aoa1 => ==(aoa1)) |> filter(:aoa2 => ==(aoa2)) |> first
        if axis == :actx 
            ll,ul = lims[[:lower_lim2, :upper_lim2]]
        elseif axis == :actz
            ll,ul = lims[[:lower_lim1, :upper_lim1]]
        end
        @show ll ul (demand + ll)/(ul+ll)
        itp(mach, aoa1, aoa2, 2*(demand + ll)/(ul+ll)-1)
    end
    #mach_kts = 
    #Interpolations.interpolate(Matrix{Float64}(dat[:, [:mach, :aoa1, :aoa2, :demand]]))
    #interpolate(Multiquadratic(), collect(Matrix{Float64}(dat[:, [:mach, :aoa1, :aoa2, :demand]])'), Float64.(dat[:, :control]))
end

function calculate_force(aoa1, aoa2, mach, ua1, ua2)
    approx = control_eff |> filter(:mach => ==(mach)) |> first
    rv = mean(norm.((body_data |> filter(:mach => ==(mach)))[:,:refvel]))
    cc1 = approx[:cl1_lin] * cosd(aoa2) + approx[:cl1_const]
    cc2 = approx[:cl2_lin] * cosd(aoa1) + approx[:cl2_const]
    l1 = rv^2 .* ua1
    l2 = rv^2 .* ua2
    return [l1, l2, cc1*(l1*approx[:cl1_scale])^2 + cc2*(l2*approx[:cl2_scale])^2] 
end

(aoa1, aoa2, mach) = (5.0, -17.5, 1.5)
actx_lut = make_inv_ctrl_lookup(inv_ctrl_map, :actx)
actz_lut = make_inv_ctrl_lookup(inv_ctrl_map, :actz)
(estx, estz) = actx_lut(mach, aoa1, aoa2, 0.391), actz_lut(mach, aoa1, aoa2, 0.0)
refpts = data |> filter(:aoa1 => ==(aoa1)) |> filter(:aoa2 => ==(aoa2)) |> filter(:mach => ==(mach))
(dist, closest_command) = findmin(norm.(collect.(eachrow(refpts[:,[:actx, :actz]])) .- ([estx, estz], )))
refpts[closest_command, :]  
calculate_force(aoa1, aoa2, mach, 0.0, 0.0)



CSV.write("control_eff.csv", control_eff)
CSV.write("control_lims.csv", control_lims)
CSV.write("control_invmap.csv", inv_ctrl_map)
CSV.write("body_fit.csv", body_data |> filter(:mach => !=(0.1)))



scatter(map((c,i)->Point3(c[[1,2]]...,i), inbounds[:, :ftf], inbounds[:, :induced]))
scatter!(map((c,i)->Point3(c...), inbounds[:, :ftf], inbounds[:, :induced]))
scatter!(map((c,i)->Point3(c...), not_insane[:, :ftf], not_insane[:, :induced]))


function points_at(data, mach, aoa1, aoa2)
    @show aoa1 aoa2
    actuation_point = data |> filter(:aoa1 => ==(aoa1)) |> 
            filter(:aoa2 => ==(aoa2))|>
            filter(:mach => ==(mach))
            forces = collect.(eachrow(select(actuation_point, :fx, :fy, :fz)))
            derot_mat = lookat_rmat(normalize(collect(actuation_point[1, [:vx, :vy, :vz]])), [0,-1,0])
            minfrc, min_drag_idx = findmin(el->el[2], ((derot_mat, ) .* forces))
            reference_point = (actuation_point[min_drag_idx,[:fx,:fy,:fz]]) |> collect
            pts = collect.((derot_mat,) .* (forces .- (reference_point,)))

            deskew = rebasis.(([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]], ), pts)
            return pts
end

# validate control approximation 
# need to reskew
begin
    aoa1 = 25.0
    aoa2 = 0.0
    mach = 1.2
    approx = control_eff |> filter(:mach => ==(mach)) |> first
    rv = mean(norm.((body_data |> filter(:mach => ==(mach)))[:,:refvel]))
    cc1 = approx[:cl1_lin] * cosd(aoa2) + approx[:cl1_const]
    cc2 = approx[:cl2_lin] * cosd(aoa1) + approx[:cl2_const]
    lims = control_lims |> filter(:mach => ==(mach)) |> filter(:aoa1 => ==(aoa1)) |> filter(:aoa2 => ==(aoa2)) |> first
    l1r = rv^2 .* LinRange(-lims[:lower_lim1],lims[:upper_lim1],20)
    l2r = rv^2 .* LinRange(-lims[:lower_lim2],lims[:upper_lim2],20)
    dvs = [cc1*(l1*approx[:cl1_scale])^2 + cc2*(l2*approx[:cl2_scale])^2 for l1 ∈ l1r, l2 ∈ l2r] 
    @show approx[:cl1_scale]
    f = Figure()
    ax = Axis3(f[1,1];xlabel="Lift X", ylabel="Lift Y", zlabel="Drag", aspect=:data, limits=((nothing, nothing),(nothing, nothing),(0.0,50.0)), zticks=[0.0,50.0])
    Makie.surface!(ax, approx[:cl1_scale] .* l1r, approx[:cl2_scale] .* l2r, dvs)
    #scatter!((adjusted[:, :ftf]))
    #scatter!((inbounds[:, :ftf]))
    scatter!(ax, (RotXY(deg2rad(90), deg2rad(90)),) .* Point3.(points_at(data |> filter(:acty => ==(0.0)), mach, aoa1, aoa2)))
    f
end

#=
scatter!((df -> Point3.(df[:, :aoa_net] ./ 45, df[:, :mach] / 3, df[:, :lift] * 100))(body_aero_out))
scatter((df -> Point3.(df[:, :aoa_net] ./ 45, df[:, :mach] / 3, df[:, :drag] * 100))(body_aero_out))
scatter((df -> Point3.(df[:, :aoa_net] ./ 45, df[:, :mach] / 3, df[:, :trq] * 100))(body_aero_out))
scatter((df -> Point2.(df[:, :mach], df[:, :lift]))(body_aero_out |> filter(:aoa_net => ==(5.0))))
=#

planarize([-0.15,0.0,1.0])
planarize(RotZX(deg2rad(-15.0), deg2rad(-25.0)) * [0.0,0.0,1.0])

norm_to_el1(v) = v/abs(v[1])
vr1 = norm_to_el1(RotMatrix(deg2rad(25)) * [1.0, 0.0])
vr2 = norm_to_el1(RotMatrix(deg2rad(25)) * [1.0, 0.0])
vr3 = [vr1[2], vr2[2], 1.0]
vel = vr3/norm(vr3)


function angle_in_plane(vec, ref, normal)
    return atand(dot(cross(vec, ref), normal), dot(vec, ref))
end

angle_in_plane(vel, [0,0,1], [0,1,0])
