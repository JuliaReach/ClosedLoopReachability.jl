# ========================================
# Projection operations
# ========================================

function _project_oa(X::AbstractLazyReachSet, vars, t; remove_zero_generators=true)
    return Project(X, vars)
end

function _project_oa(X::AbstractTaylorModelReachSet, vars, t; remove_zero_generators=true)
    Z = overapproximate(X, Zonotope, t, remove_zero_generators=remove_zero_generators)

    πZ = concretize(Projection(set(Z), vars))
    if remove_zero_generators
        πZ = LazySets.remove_zero_generators(πZ)
    end
    return ReachSet(πZ, tspan(X))
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

# ==============================================================
# Construct the initial states for the continuous post-operator
# ==============================================================

using ReachabilityAnalysis: TaylorModel1, TaylorModelN, fp_rpa, zeroBox, symBox

abstract type AbstractReconstructionMethod end

struct CartesianProductReconstructor <: AbstractReconstructionMethod end

function _reconstruct(method::CartesianProductReconstructor, P₀::LazySet, U₀::LazySet, X, ti)
    Q₀ = P₀ × U₀
    return Q₀
end

function _reconstruct(method::CartesianProductReconstructor, P₀::LazySet, U₀::Vector{<:LazySet}, X, ti)
    @assert length(U₀) == 1 "expected the length of U₀ to be 1, got $(lenght(U₀))"
    return _reconstruct(method, P₀, first(U₀), X, ti)
end

struct TaylorModelReconstructor <: AbstractReconstructionMethod end

# if no Taylor model is available => use the given set P₀
function _reconstruct(method::TaylorModelReconstructor, P₀::LazySet, U₀, X::Nothing, ti) where {N}
    return _reconstruct(CartesianProductReconstructor(), P₀, U₀, X, ti)
end

function _reconstruct(method::TaylorModelReconstructor, P₀::LazySet, U₀::Vector{<:LazySet}, X::TaylorModelReachSet{N}, ti) where {N}
    @assert length(U₀) == 1 "expected the length of U₀ to be 1, got $(lenght(U₀))"
    return _reconstruct(method, P₀, first(U₀), X, ti)
end

function _reconstruct(method::TaylorModelReconstructor, P₀::LazySet, U₀::LazySet, X::TaylorModelReachSet{N}, ti) where {N}
    # evaluate X at the final time of the reach-set
    S = set(X)
    tn = sup(domain(X)) # assume that the reach set spans the whole period (checked outside this method)
    X_Δt = evaluate(S, tn)

    n = dim(P₀) # number of state variables
    m = dim(U₀) # control variables

    vTM = Vector{TaylorModel1{TaylorN{N}, N}}(undef, n + m)

    # construct state variables
    orderT = get_order(first(set(X)))
    orderQ = get_order(X_Δt[1])

    zeroI = interval(zero(N), zero(N))
    Δtn = zeroI
    for i in 1:n
        rem = remainder(S[i])
        W = TaylorModelN(X_Δt[i], rem, zeroBox(n + m), symBox(n + m))
        Ŵ = fp_rpa(W)
        p = Taylor1(TaylorN(polynomial(Ŵ)), orderT)
        vTM[i] = TaylorModel1(p, zeroI, zeroI, Δtn)
    end

    # fill the components for the inputs
    @assert dim(U₀) == 1
    @assert m == 1
    U₀ = overapproximate(U₀, Interval)
    I = U₀.dat
    pi = mid(I) + zero(TaylorN(n+m, order=orderQ))
    d = diam(I) / 2
    rem = interval(-d, d)
    @inbounds vTM[n+m] = TaylorModel1(Taylor1(pi, orderT), rem, zeroI, Δtn)

    return TaylorModelReachSet(vTM, Δtn)
end
