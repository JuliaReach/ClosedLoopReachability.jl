# ========================================
# Projection operations
# ========================================

function _Projection(X::AbstractLazyReachSet, st_vars)
    ReachabilityAnalysis.Projection(X, st_vars)
end

function _Projection(X::AbstractTaylorModelReachSet, st_vars)
    Z = overapproximate(X, Zonotope)
    ReachabilityAnalysis.Projection(Z, st_vars)
end

# ========================================
# Decomposition operations
# ========================================

# decompose a set into the Cartesian product of intervals
using LazySets.Arrays: SingleEntryVector

function _decompose_1D(X0::LazySet{N}) where {N}
    n = dim(X0)
    out = Vector{Interval{Float64}}(undef, n)

    @inbounds for i in 1:n
        eᵢ = SingleEntryVector(i, n, one(N))
        out[i] = Interval(-ρ(-eᵢ, X0), ρ(eᵢ, X0))
    end
    return CartesianProductArray(out)
end
