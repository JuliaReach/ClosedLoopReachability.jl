"""
    read_POLAR(filename::String)

Read a neural network stored in POLAR format.

### Input

- `filename` -- name of the POLAR file

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

The POLAR format uses the same parameter format as Sherlock (see
[`read_Sherlock`](@ref)) but allows for general activation functions.
"""
function read_POLAR(filename::String)
    layers = nothing
    open(filename, "r") do io
        n_inputs = parse(Int, readline(io))  # number of neurons in input layer
        n_outputs = parse(Int, readline(io))  # number of neurons in output layer
        # number of layer operations (+1 because the file stores the number of
        # hidden layers)
        n_layer_ops = parse(Int, readline(io)) + 1

        # number of neurons per layer
        n_neurons = Vector{Int}(undef, n_layer_ops + 1)
        @inbounds begin
            n_neurons[1] = n_inputs
            # one line for each number of neurons in the hidden layers
            for i in 2:n_layer_ops
                n_neurons[i] = parse(Int, readline(io))
            end
            n_neurons[end] = n_outputs
        end

        # one line for each activation function
        activations = [available_activations[readline(io)] for _ in 1:n_layer_ops]

        T = DenseLayerOp{<:ActivationFunction, Matrix{Float32}, Vector{Float32}}
        layers = Vector{T}(undef, n_layer_ops)

        # the layers use the Sherlock format
        @inbounds for i in 1:n_layer_ops
            W, b = _read_layer_Sherlock(io, n_neurons[i+1], n_neurons[i])
            layers[i] = DenseLayerOp(W, b, activations[i])
        end
    end

    return FeedforwardNetwork(layers)
end
