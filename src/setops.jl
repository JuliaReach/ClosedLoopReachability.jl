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
