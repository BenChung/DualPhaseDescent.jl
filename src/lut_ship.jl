function (a::AeroLookupTs)(tx::T, ty::T) where T <: (SparseConnectivityTracer).GradientTracer
    return @noinline((SparseConnectivityTracer).gradient_tracer_2_to_1(tx, ty, false, false))
end
function (a::AeroLookupTs)(dx::D, dy::D) where {P, T <: (SparseConnectivityTracer).GradientTracer, D <: (SparseConnectivityTracer).Dual{P, T}}
    x = (SparseConnectivityTracer).primal(dx)
    y = (SparseConnectivityTracer).primal(dy)
    p_out = a(x, y)
    tx = (SparseConnectivityTracer).tracer(dx)
    ty = (SparseConnectivityTracer).tracer(dy)
    is_der1_arg1_zero = (SparseConnectivityTracer).is_der1_arg1_zero_local(a, x, y)
    is_der1_arg2_zero = (SparseConnectivityTracer).is_der1_arg2_zero_local(a, x, y)
    t_out = @noinline((SparseConnectivityTracer).gradient_tracer_2_to_1(tx, ty, is_der1_arg1_zero, is_der1_arg2_zero))
    return (SparseConnectivityTracer).Dual(p_out, t_out)
end
function (a::AeroLookupTs)(tx::(SparseConnectivityTracer).GradientTracer, ::Real)
    return @noinline((SparseConnectivityTracer).gradient_tracer_1_to_1(tx, false))
end
function (a::AeroLookupTs)(::Real, ty::(SparseConnectivityTracer).GradientTracer)
    return @noinline((SparseConnectivityTracer).gradient_tracer_1_to_1(tx, false))
end             
function (a::AeroLookupTs)(dx::D, y::Real) where {P, T <: (SparseConnectivityTracer).GradientTracer, D <: (SparseConnectivityTracer).Dual{P, T}}
    x = (SparseConnectivityTracer).primal(dx)
    p_out = a(x, y)
    tx = (SparseConnectivityTracer).tracer(dx)
    is_der1_arg1_zero = (SparseConnectivityTracer).is_der1_arg1_zero_local(a, x, y)
    t_out = @noinline((SparseConnectivityTracer).gradient_tracer_1_to_1(tx, is_der1_arg1_zero))
    return (SparseConnectivityTracer).Dual(p_out, t_out)
end
function (a::AeroLookupTs)(x::Real, dy::D) where {P, T <: (SparseConnectivityTracer).GradientTracer, D <: (SparseConnectivityTracer).Dual{P, T}}
    y = (SparseConnectivityTracer).primal(dy)
    p_out = a(x, y)
    ty = (SparseConnectivityTracer).tracer(dy)
    is_der1_arg2_zero = (SparseConnectivityTracer).is_der1_arg2_zero_local(a, x, y)
    t_out = @noinline((SparseConnectivityTracer).gradient_tracer_1_to_1(ty, is_der1_arg2_zero))
    return (SparseConnectivityTracer).Dual(p_out, t_out)
end
function (a::AeroLookupTs)(tx::T, ty::T) where T <: (SparseConnectivityTracer).HessianTracer
    return @noinline((SparseConnectivityTracer).hessian_tracer_2_to_1(tx, ty, false, false, false, false, false))
end         
function (a::AeroLookupTs)(dx::D, dy::D) where {P, T <: (SparseConnectivityTracer).HessianTracer, D <: (SparseConnectivityTracer).Dual{P, T}}
    x = (SparseConnectivityTracer).primal(dx)
    y = (SparseConnectivityTracer).primal(dy)
    p_out = a(x, y)
    tx = (SparseConnectivityTracer).tracer(dx)
    ty = (SparseConnectivityTracer).tracer(dy)
    is_der1_arg1_zero = (SparseConnectivityTracer).is_der1_arg1_zero_local(a, x, y)
    is_der2_arg1_zero = (SparseConnectivityTracer).is_der2_arg1_zero_local(a, x, y)
    is_der1_arg2_zero = (SparseConnectivityTracer).is_der1_arg2_zero_local(a, x, y)
    is_der2_arg2_zero = (SparseConnectivityTracer).is_der2_arg2_zero_local(a, x, y)
    is_der_cross_zero = (SparseConnectivityTracer).is_der_cross_zero_local(a, x, y)
    t_out = @noinline((SparseConnectivityTracer).hessian_tracer_2_to_1(tx, ty, is_der1_arg1_zero, is_der2_arg1_zero, is_der1_arg2_zero, is_der2_arg2_zero, is_der_cross_zero))
    return (SparseConnectivityTracer).Dual(p_out, t_out)
end
function (a::AeroLookupTs)(tx::(SparseConnectivityTracer).HessianTracer, y::Real)
    return @noinline((SparseConnectivityTracer).hessian_tracer_1_to_1(tx, false, false))
end
function (a::AeroLookupTs)(x::Real, ty::(SparseConnectivityTracer).HessianTracer)
    return @noinline((SparseConnectivityTracer).hessian_tracer_1_to_1(ty, false, false))
end
function (a::AeroLookupTs)(dx::D, y::Real) where {P, T <: (SparseConnectivityTracer).HessianTracer, D <: (SparseConnectivityTracer).Dual{P, T}}
    x = (SparseConnectivityTracer).primal(dx)
    p_out = a(x, y)
    tx = (SparseConnectivityTracer).tracer(dx)
    is_der1_arg1_zero = (SparseConnectivityTracer).is_der1_arg1_zero_local(a, x, y)
    is_der2_arg1_zero = (SparseConnectivityTracer).is_der2_arg1_zero_local(a, x, y)
    t_out = @noinline((SparseConnectivityTracer).hessian_tracer_1_to_1(tx, is_der1_arg1_zero, is_der2_arg1_zero))
    return (SparseConnectivityTracer).Dual(p_out, t_out)
end
function (a::AeroLookupTs)(x::Real, dy::D) where {P, T <: (SparseConnectivityTracer).HessianTracer, D <: (SparseConnectivityTracer).Dual{P, T}}
    y = (SparseConnectivityTracer).primal(dy)
    p_out = a(x, y)
    ty = (SparseConnectivityTracer).tracer(dy)
    is_der1_arg2_zero = (SparseConnectivityTracer).is_der1_arg2_zero_local(a, x, y)
    is_der2_arg2_zero = (SparseConnectivityTracer).is_der2_arg2_zero_local(a, x, y)
    t_out = @noinline((SparseConnectivityTracer).hessian_tracer_1_to_1(ty, is_der1_arg2_zero, is_der2_arg2_zero))
    return (SparseConnectivityTracer).Dual(p_out, t_out)
end