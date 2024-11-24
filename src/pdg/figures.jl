# FIGURE: costev.png
rch_min_fuel = CSV.read("src/pdg/figure_data/rch_min_fuel.csv")
rch_min_time = CSV.read("src/pdg/figure_data/rch_min_time.csv")

f = Figure()
relcosts_min_fuel = rch_min_fuel .- minimum(rch_min_fuel) .+ 1e-5
relcosts_min_time = rch_min_time .- minimum(rch_min_time) .+ 1e-5
ax = Makie.Axis(f[1,1], xlabel="Iteration", ylabel="Nonlinear cost vs. converged solution", yscale=log10,limits=((0,30),(1e-6, 50*max(maximum(relcosts_min_time), maximum(relcosts_min_fuel)))))
min_fuel = lines!(ax, relcosts_min_fuel)
min_time = lines!(ax, relcosts_min_time) 
axislegend(ax, [min_fuel, min_time], ["Min-fuel", "Min-time"])