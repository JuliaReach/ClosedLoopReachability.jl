# NOTE: `layers` actually contains the layer transformations, so the number of
# layers is `length(layers) + 1`.
struct FeedforwardNetwork{L} <: AbstractNeuralNetwork
    layers::L
end

(N::FeedforwardNetwork)(x) = reduce((a1, a2) -> a2∘a1, N.layers)(x)

function Base.:(==)(N1::FeedforwardNetwork, N2::FeedforwardNetwork)
    return N1.layers == N2.layers
end

dim_in(N::FeedforwardNetwork) = dim_in(first(N.layers))

dim_out(N::FeedforwardNetwork) = dim_out(last(N.layers))
