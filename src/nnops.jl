# ================================================
# Intrenal forward network functions
# ================================================

# fallback to NV
function _forward_network(solver::AbstractSolver, network::Network, X::LazySet)
    NV.forward_network(solver, network, X)
end

function _forward_network(solver::BaB, network::Network, X::AbstractPolytope)
    sol = NV.solve(solver, NV.Problem(network, X, Interval(-Inf, Inf)))
    first(sol.reachable)
end

function _forward_network(solver::Sherlock, network::Network, X::AbstractPolytope)
    sol = NV.solve(solver, NV.Problem(network, X, Interval(-Inf, Inf)))
    first(sol.reachable)
end

# ================================================
# Zonotope-based bounding
# ================================================
struct ZonotopeBounder <: NV.AbstractSolver
#
end

function _forward_network(solver::ZonotopeBounder, network::Network, X::LazySet)
    Z = ReachabilityAnalysis._convert_or_overapproximate(Zonotope, X)
    _forward_network_zono(network, Z)
end

function _forward_network_zono(network::Network, X::Zonotope)
    nlayers = length(network.layers) # TODO getter function ?
    Z = copy(X)
    @inbounds for i in 1:nlayers
        layer = network.layers[i]
        W = layer.weights
        b = layer.bias
        Z = _forward_layer_zono(W, b, layer.activation, Z)
    end
    return remove_zero_generators(Z)
end

function _forward_layer_zono(W::AbstractMatrix, b::AbstractVector, ::ReLU, Z::Zonotope)
    Y = affine_map(W, Z, b)
    return relu(Y)
end

function _forward_layer_zono(W::AbstractMatrix, b::AbstractVector, ::Id, Z::Zonotope)
    return affine_map(W, Z, b)
end
