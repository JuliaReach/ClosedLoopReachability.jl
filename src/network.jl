struct Layer{F<:ActivationFunction, N<:Real}
    weights::Matrix{N}
    bias::Vector{N}
    activation::F
end

abstract type AbstractNetwork end

struct Network <: AbstractNetwork
    layers::Vector{Layer}
end

Base.length(L::Layer) = length(L.bias)
