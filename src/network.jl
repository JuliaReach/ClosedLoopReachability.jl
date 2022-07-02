struct Layer{F<:ActivationFunction, N<:Real}
    weights::Matrix{N}
    bias::Vector{N}
    activation::F
end

Base.length(L::Layer) = length(L.bias)

function Base.:(==)(L1::Layer, L2::Layer)
    return L1.weights == L2.weights &&
           L1.bias == L2.bias &&
           L1.activation == L2.activation
end


abstract type AbstractNetwork end

struct Network <: AbstractNetwork
    layers::Vector{Layer}
end

function Base.:(==)(N1::Network, N2::Network)
    return N1.layers == N2.layers
end
