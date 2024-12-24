model = Model(Clarabel.Optimizer)
@variable(model, x[1:5])

set_optimizer_attribute(model, "verbose", false)
set_optimizer_attribute(model, "presolve_enable", false)
set_optimizer_attribute(model, "chordal_decomposition_enable", false)
lin = rand(5)
cst = rand(5)
cst[1] = 1

cs = MOI.add_constraint(model.moi_backend, MOI.VectorAffineFunction([
    MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1.0, x[1].index))
    MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, x[2].index))
    MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, x[3].index))
    MOI.VectorAffineTerm(4, MOI.ScalarAffineTerm(1.0, x[4].index))
    MOI.VectorAffineTerm(5, MOI.ScalarAffineTerm(1.0, x[5].index))
], -ones(5)), MOI.Nonnegatives(5))
@objective(model, Min, 10*sum(x .* x))
optimize!(model)

function modme(model, cis, x)
    for i=1:1000
        for i=1:1
            MOI.modify(model.moi_backend, cs, MOI.MultirowChange(x[1].index, [(2, -1.0)]))
            MOI.modify(model.moi_backend, cs, MOI.VectorConstantChange(-2 * ones(5)))
        end
        optimize!(model)
    end
end

@profview modme(model, cis, x)



function MOI.Utilities._modify_coefficients(
    terms::Vector{MOI.VectorAffineTerm{T}},
    variable::MOI.VariableIndex,
    new_coefficients::Vector{Tuple{Int64,T}},
) where {T}
    # establish ordering invariant in the backing terms
    sort!(terms, 
        by=(t) -> (t.output_index, t.scalar_term.variable.value))
    sort!(new_coefficients, lt=(t1,t2) -> t1[1] < t2[1])

    iterm = 1
    icoeff = 1
    while iterm < length(terms)
        if icoeff > length(new_coefficients) break end
        nc = new_coefficients[icoeff]
        if terms[iterm].output_index < nc[1]
            iterm += 1
            continue
        elseif terms[iterm].output_index > nc[1]
            insert!(terms, iterm, MOI.VectorAffineTerm(
                nc[1],
                MOI.ScalarAffineTerm(nc[2], variable),
            ))
            iterm += 1
            icoeff += 1
            continue
        end
        if terms[iterm].scalar_term.variable.value < variable.value
            iterm += 1
            continue
        elseif terms[iterm].scalar_term.variable.value > variable.value
            insert!(terms, iterm, MOI.VectorAffineTerm(
                nc[1],
                MOI.ScalarAffineTerm(nc[2], variable)
            ))
            icoeff += 1
            iterm += 1
            continue
        end
        terms[iterm] = MOI.VectorAffineTerm(
            nc[1],
            MOI.ScalarAffineTerm(nc[2], variable)
        )
        icoeff += 1
        iterm += 1
    end
    return
end