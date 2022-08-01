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
