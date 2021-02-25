# ========================================
# Projection operations
# ========================================

function _project_oa(X::AbstractLazyReachSet, vars)
    return Project(X, vars)
end

function _project_oa(X::AbstractTaylorModelReachSet, vars)
    Z = overapproximate(X, Zonotope)
    return project(Z, vars)
end

# ========================================
# Decomposition operations
# ========================================

# decompose a set into the Cartesian product of intervals
using LazySets.Arrays: SingleEntryVector

function _decompose_1D(X0::LazySet{N}) where {N}
    n = dim(X0)
    out = Vector{Interval{Float64, IA.Interval{Float64}}}(undef, n)

    @inbounds for i in 1:n
        eᵢ = SingleEntryVector(i, n, one(N))
        out[i] = Interval(-ρ(-eᵢ, X0), ρ(eᵢ, X0))
    end
    return CartesianProductArray(out)
end
