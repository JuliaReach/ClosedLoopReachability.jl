struct FeedforwardNetwork{L} <: AbstractNeuralNetwork
    layers::L
end

function Base.:(==)(N1::FeedforwardNetwork, N2::FeedforwardNetwork)
    return N1.layers == N2.layers
end
