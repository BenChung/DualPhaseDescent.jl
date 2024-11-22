using Polyhedra, QHull, LazyArrays


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