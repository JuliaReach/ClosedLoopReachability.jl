# ===========================================
# Activation functions on sets or reach-sets
# ===========================================

@inline _bounds(H, i) = (H.center[i] - H.radius[i], H.center[i] + H.radius[i])

relu(Z::Zonotope) = relu!(copy(Z))

# proof-of-concept implementation of Theorem 3.1 in
# Fast and Effective Robustness Certification,
# G. Singh, T. Gehr, M. Mirman, M. Püschel, M. Vechev
# https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf
function relu!(Z::Zonotope{N}) where {N}
    c = Z.center
    G = Z.generators
    n, m = size(G)
    H = overapproximate(Z, Hyperrectangle)
    Gnew = zeros(N, n, n)

    @inbounds for i in 1:n
        lx, ux = _bounds(H, i)
        if !_leq(lx, zero(N))
            nothing
        elseif _leq(ux, zero(N)) || isapproxzero(lx)
            c[i] = zero(N)
            G[i, :] = zeros(N, m)
        else
            λ = ux / (ux - lx)
            μ = - λ * lx / 2
            c[i] = c[i] * λ + μ
            G[i, :] = G[i, :] .* λ
            Gnew[i, i] = μ
        end
    end
    Z = Zonotope(c, hcat(G, Gnew))
    return remove_zero_generators(Z)
end

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
