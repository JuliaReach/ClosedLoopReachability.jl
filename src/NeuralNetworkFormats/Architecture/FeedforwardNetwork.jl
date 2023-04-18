"""
    FeedforwardNetwork{L} <: AbstractNeuralNetwork

Standard implementation of a feedforward neural network which stores the layer
operations.

### Fields

- `layers` -- vector of layer operations (see [`AbstractLayerOp`](@ref))

### Notes

The field `layers` contains the layer operations, so the number of layers is
`length(layers) + 1`.
"""
struct FeedforwardNetwork{L} <: AbstractNeuralNetwork
    layers::L

    function FeedforwardNetwork(layers::L; validate=Val(true)) where {L}
        if validate isa Val{true}
            i = _first_inconsistent_layer(layers)
            i != 0 && throw(ArgumentError("inconsistent layer dimensions at " *
                                          "index $i"))
        end

        return new{L}(layers)
    end
end

function _first_inconsistent_layer(L)
    prev = nothing
    for (i, l) in enumerate(L)
        if !isnothing(prev) && dim_in(l) != dim_out(prev)
            return i
        end
        prev = l
    end
    return 0
end

(N::FeedforwardNetwork)(x) = reduce((a1, a2) -> a2∘a1, N.layers)(x)

function Base.:(==)(N1::FeedforwardNetwork, N2::FeedforwardNetwork)
    return N1.layers == N2.layers
end

dim_in(N::FeedforwardNetwork) = dim_in(first(N.layers))

dim_out(N::FeedforwardNetwork) = dim_out(last(N.layers))
