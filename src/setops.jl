# ========================================
# Projection operations
# ========================================

function _project_oa(R::AbstractLazyReachSet, vars, ::Any; remove_zero_generators=true)
    return project(R, vars)
end

function _project_oa(R::AbstractTaylorModelReachSet, vars, t; remove_zero_generators=true)
    Z = overapproximate(R, Zonotope, t)
    return LazySets.project(set(Z), vars; remove_zero_generators=remove_zero_generators)
end

# ==============================================================
# Construct the initial states for the continuous post-operator
# ==============================================================

abstract type AbstractReconstructionMethod end

struct ConcreteCartesianProductReconstructor <: AbstractReconstructionMethod end

function _reconstruct(::ConcreteCartesianProductReconstructor, P₀::LazySet, U₀::LazySet, R, ti)
    Q₀ = cartesian_product(P₀, U₀)
    return Q₀
end

struct LazyCartesianProductReconstructor <: AbstractReconstructionMethod end

function _reconstruct(::LazyCartesianProductReconstructor, P₀::LazySet, U₀::LazySet, R, ti)
    Q₀ = P₀ × U₀
    return Q₀
end

@with_kw struct TaylorModelReconstructor <: AbstractReconstructionMethod
    box::Bool = true
end

# if no Taylor model is available => use the given set P₀
function _reconstruct(::TaylorModelReconstructor, P₀::LazySet, U₀, R::Nothing, ti)
    return _reconstruct(ConcreteCartesianProductReconstructor(), P₀, U₀, R, ti)
end

function _reconstruct(method::TaylorModelReconstructor, P₀::LazySet, U₀::Vector{<:LazySet},
                      R::TaylorModelReachSet{N}, ti) where {N}
    @assert length(U₀) == 1 "expected the length of U₀ to be 1, got $(length(U₀))"
    return _reconstruct(method, P₀, first(U₀), R, ti)
end

function _reconstruct(method::TaylorModelReconstructor, P₀::LazySet, U₀::LazySet,
                      R::TaylorModelReachSet{N}, ::Any) where {N}
    # evaluate R at the final time of the reach-set
    S = set(R)
    tn = sup(domain(R)) # assume that the reach set spans the whole period (checked outside this method)
    X_Δt = evaluate(S, tn)

    n = dim(P₀) # number of state variables = dim(P₀) - dim(U₀)
    m = dim(U₀) # control variables

    vTM = Vector{TaylorModel1{TaylorN{N},N}}(undef, n + m)

    # construct state variables
    orderT = get_order(first(S))
    orderQ = get_order(X_Δt[1])

    zeroI = interval(zero(N), zero(N))
    Δtn = zeroI
    @inbounds for i in 1:n
        W = TaylorModelN(X_Δt[i], zeroI, zeroBox(n + m), symBox(n + m))
        Ŵ = fp_rpa(W)
        p = Taylor1(TaylorN(polynomial(Ŵ)), orderT)
        rem = remainder(Ŵ)
        vTM[i] = TaylorModel1(p, rem, zeroI, Δtn)
    end

    # fill the components for the inputs
    if method.box || isa(U₀, AbstractHyperrectangle)
        B₀ = convert(IntervalBox, box_approximation(U₀))
        @inbounds for i in 1:m
            I = B₀[i]
            p = mid(I) + zero(TaylorN(n + m; order=orderQ))
            d = diam(I) / 2
            rem = interval(-d, d)
            vTM[n + i] = TaylorModel1(Taylor1(p, orderT), rem, zeroI, Δtn)
        end
    else
        Z₀ = ReachabilityAnalysis._convert_or_overapproximate(U₀, Zonotope)
        Z₀ = ReachabilityAnalysis._reduce_order(Z₀, 2; force_reduction=true)
        # NOTE if we used _overapproximate_structured directly :
        # Utm₀ = set(ReachabilityAnalysis._overapproximate_structured(Z₀, TaylorModelReachSet, orderT=orderT, orderQ=orderQ))
        # @inbounds for i in 1:m
        #     vTM[n+i] = Utm₀[i]
        # end

        x = set_variables("x"; numvars=n + m, order=orderQ)
        xc = view(x, (n + 1):(n + m))
        G = Z₀.generators
        c = Z₀.center
        @assert size(G) == (m, 2m)
        M = view(G, :, 1:m)
        D = view(G, :, (m + 1):(2m))
        @assert isdiag(D)
        @inbounds for i in 1:m
            p = c[i] + sum(view(M, i, :) .* xc)
            d = abs(D[i, i])
            rem = interval(-d, d)
            vTM[n + i] = TaylorModel1(Taylor1(p, orderT), rem, zeroI, Δtn)
        end
    end
    return TaylorModelReachSet(vTM, Δtn)
end
