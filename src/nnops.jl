# ================================================
# Internal forward network functions
# ================================================

#=

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

# This is now Ai2z
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
    return overapproximate(Rectification(Y), Zonotope)
end

function _forward_layer_zono(W::AbstractMatrix, b::AbstractVector, ::Id, Z::Zonotope)
    return affine_map(W, Z, b)
end
=#

# ================================================
# Extension of NeuralVerification structs
# ================================================

using NeuralVerification: ActivationFunction

"""
    Sigmoid <: ActivationFunction
    (Sigmoid())(x) -> 1 ./ (1 .+ exp.(-x))
"""
struct Sigmoid <: ActivationFunction end

"""
    Tanh <: ActivationFunction
    (Tanh())(x) -> tanh.(x)
"""
struct Tanh <: ActivationFunction end

(f::Sigmoid)(x) = @. 1 / (1 + exp(-x))
(f::Tanh)(x) = tanh.(x)

# ================================================
# Reading a network in YAML format
# (load the data with YAML.jl)
# ================================================

const ACT_YAML = Dict("Sigmoid"=>Sigmoid(), "Tanh"=>Tanh(), "Id"=>Id(), "ReLU"=>ReLU())

function read_yaml(data::Dict)
    NLAYERS = length(data["offsets"])
    layers = []
    for k in 1:NLAYERS
        weights = data["weights"][k]
        W = copy(reduce(hcat, weights)')
        b = data["offsets"][k]
        a = ACT_YAML[data["activations"][k]]
        L = Layer(W, b, a)
        push!(layers, L)
    end
    return Network(layers)
end
