# FIGURE: costev.png
rch_min_fuel = CSV.read("src/pdg/figure_data/rch_min_fuel.csv", DataFrame)[:,:rch]
rch_min_time = CSV.read("src/pdg/figure_data/rch_min_time.csv", DataFrame)[:,:rch]

f = Figure()
relcosts_min_fuel = rch_min_fuel .- minimum(rch_min_fuel) .+ 1e-5
relcosts_min_time = rch_min_time .- minimum(rch_min_time) .+ 1e-5
ax = Makie.Axis(f[1,1], xlabel="Iteration", ylabel="Nonlinear cost vs. converged solution", yscale=log10,limits=((0,30),(1e-6, 50*max(maximum(relcosts_min_time), maximum(relcosts_min_fuel)))))
min_fuel = lines!(ax, relcosts_min_fuel)
min_time = lines!(ax, relcosts_min_time) 
axislegend(ax, [min_fuel, min_time], ["Min-fuel", "Min-time"])
save("spbm_convplot.pdf", f; size=(400,400))

#figure: lift_csc.pdf
highres_aoa = CSV.read("src/pdg/figure_data/highres_aoa.csv", DataFrame)
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
    aero_data = aero_data |> filter(:aoa1 => (a -> (a .>= 1.0) .| (a .== 0.0)))
    for expt in eachrow(aero_data)
        refvel = norm([expt[:vx], expt[:vy], expt[:vz]])
        aoa1, aoa2, mach = expt[[:aoa1, :aoa2, :mach]]
        Rbi = RotXZ(deg2rad(-aoa1), deg2rad(-aoa2))
        frc_windframe = Rbi * collect(expt[[:fx, :fy, :fz]])
        trq_windframe = Rbi * collect(expt[[:tx, :ty, :tz]])
        bodydir = Rbi * [0, 1.0, 0] 
        up = [0.0, 1.0, 0.0]
        bv1 = cross(up, bodydir)
        bv1 = bv1 / norm(bv1)
        bv2 = cross(up, bv1 / norm(bv1))
        aa_rbi = AngleAxis(Rbi)
        drag = dot(up, frc_windframe) / refvel^2
        frc_err = dot(bv1, frc_windframe)
        lift = csc(aa_rbi.theta) * dot(bv2, frc_windframe) / refvel^2
        trq = csc(aa_rbi.theta) * dot(bv1, trq_windframe) / refvel^2
        #trq_err = norm(trq_windframe - trq*bv1)/trq
        push!(body_aero, (mach=mach, aoa1=aoa1, aoa2=aoa2, aoa_net=round(rad2deg(aa_rbi.theta); digits=8), refvel = refvel,
        lift_raw=dot(bv2, frc_windframe), drag_raw = dot(up, frc_windframe), trq_raw = dot(bv1, trq_windframe),
        lift=isnan(lift) ? 0.0 : lift, drag=isnan(drag) ? 0.0 : drag, frc_err=frc_err / drag, trq=isnan(trq) ? 0.0 : trq))
    end
    return combine(groupby(select(body_aero, Not([:aoa1, :aoa2])), [:mach, :aoa_net]), 
    :lift => mean => :lift, :drag => mean => :drag, :trq => mean => :trq, 
    :lift_raw => mean => :lift_raw, :drag_raw => mean => :drag_raw, :trq_raw => mean => :trq_raw,
    :refvel => mean => :refvel)
end



zeronan(l; av=0.0) = ifelse(isnan(l), av, l)
bdf = transform(make_body_aero(highres_aoa), 
    :lift_raw => (l -> zeronan.(l)) => :lift_raw,
    :trq_raw => (l -> zeronan.(l)) => :trq_raw)
xformed = transform(bdf,
    [:lift_raw, :aoa_net] => ((lift, aoa) -> zeronan.(lift .* cscd.(aoa))) => :xform_lift,
    [:trq_raw, :aoa_net] => ((trq, aoa) -> zeronan.(trq .* cscd.(aoa))) => :xform_trq
)

function make_interp(tab, field)
    field_df = unstack(tab, :mach, :aoa_net, field)
    aoa_kts = parse.(Float64, names(field_df)[2:end])
    mach_kts = field_df[:,:mach]
    return extrapolate(interpolate((mach_kts, aoa_kts), convert(Matrix{Float64}, Matrix(field_df[:,2:end])), Gridded(Linear())), Flat())
end

relative_error = let 
    mach = 2.0
    aoa_range = 10 .^ (-6:1e-2:1)
    xform_itp = make_interp(xformed, :xform_lift)
    raw_itp = make_interp(xformed, :lift_raw)
    f = Figure()
    ax = Makie.Axis(f[1,1], xlabel="Angle of attack (°)", ylabel="Modified to unmodified error ratio") 
    Makie.lines!(ax, aoa_range,(xform_itp.(mach, aoa_range) .* sind.(aoa_range)) ./ raw_itp.(mach, aoa_range))
    f
end
save("approx_relerror.pdf", relative_error; size=(400,400))

abs_error = let 
    mach = 2.0
    aoa_range = 10 .^ (-6:1e-2:1)
    xform_itp = make_interp(xformed, :xform_lift)
    raw_itp = make_interp(xformed, :lift_raw)
    f = Figure()
    ax = Makie.Axis(f[1,1], xlabel="Angle of attack (°)", ylabel="Predicted lift (kN)") 
    xp = Makie.lines!(ax, aoa_range,(xform_itp.(mach, aoa_range) .* sind.(aoa_range)))
    op = Makie.lines!(ax, aoa_range, raw_itp.(mach, aoa_range))
    axislegend(ax, [op, xp], ["Unmodified table", "Modified table"], position=:lt)
    f
end
save("approx_abserror.pdf", abs_error; size=(400,400))
m09 = xformed |> filter(:mach => ==(0.9))
Makie.lines(m09[:, :aoa_net], m09[:, :xform_lift])