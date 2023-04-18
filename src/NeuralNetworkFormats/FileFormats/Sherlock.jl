"""
    read_Sherlock(filename::String)

Read a neural network stored in
[Sherlock](https://github.com/souradeep-111/sherlock/blob/bf34fb4713e5140b893c98382055fb963230d69d/sherlock-network-format.pdf)
format.

### Input

- `filename` -- name of the Sherlock file

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

All layers including the output layer implicitly use a ReLU activation function.
"""
function read_Sherlock(filename::String)
    # activation functions are not read from file because they are always ReLU
    read_activations(io, n_layer_ops) = i -> ReLU()

    layer_type = DenseLayerOp{ReLU, Matrix{Float32}, Vector{Float32}}

    return _read_Sherlock_POLAR(filename, read_activations, layer_type)
end

# common code for Sherlock and POLAR format
# (the only difference is that Sherlock always uses ReLU)
function _read_Sherlock_POLAR(filename::String, read_activations, layer_type)
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

        layers = Vector{layer_type}(undef, n_layer_ops)

        activations = read_activations(io, n_layer_ops)

        # one line for each weight and bias in the following order:
        # - all incoming weights to neuron 1 in layer 1
        # - bias term of neuron 1 in layer 1
        # - all incoming weights to neuron 2 in layer 1
        # - bias term of neuron 2 in layer 1
        # - ...
        # - all incoming weights to the last neuron in layer 1
        # - bias term of the last neuron in layer 1
        # continue with layer 2 until the output layer
        @inbounds for i in 1:n_layer_ops
            W, b = _read_layer_Sherlock(io, n_neurons[i+1], n_neurons[i])
            layers[i] = DenseLayerOp(W, b, activations(i))
        end
    end

    return FeedforwardNetwork(layers)
end

function _read_layer_Sherlock(io, m, n)
    W = Matrix{Float32}(undef, m, n)
    b = Vector{Float32}(undef, m)
    @inbounds for i in 1:m
        for j in 1:n
            W[i, j] = parse(Float32, readline(io))
        end
        b[i] = parse(Float32, readline(io))
    end
    return W, b
end

"""
    write_Sherlock(N::FeedforwardNetwork, filename::String)

Write a neural network to a file in
[Sherlock](https://github.com/souradeep-111/sherlock/blob/bf34fb4713e5140b893c98382055fb963230d69d/sherlock-network-format.pdf)
format.

### Input

- `N`        -- feedforward neural network
- `filename` -- name of the output file

### Output

`nothing`. The network is written to the output file.

### Notes

The Sherlock format requires that all activation functions are ReLU.
"""
function write_Sherlock(N::FeedforwardNetwork, filename::String)
    n_inputs = dim_in(N)
    n_outputs = dim_out(N)
    n_hidden_layers = length(N.layers) - 1
    open(filename, "w") do io
        println(io, string(n_inputs))  # number of neurons in input layer
        println(io, string(n_outputs))  # number of neurons in output layer
        println(io, string(n_hidden_layers))  # number of hidden layers

        # one line for each number of neurons in the hidden layers
        @inbounds for i in 1:n_hidden_layers
            println(io, string(dim_out(N.layers[i])))
        end

        # one line for each weight and bias of the hidden and output layers
        @inbounds for layer in N.layers
            _write_layer_Sherlock(io, layer)
        end
    end
    nothing
end

function _write_layer_Sherlock(io, layer)
    @assert layer.activation isa ReLU "the Sherlock format requires ReLU " *
        "activations everywhere, but the network contains a " *
        "`$(typeof(layer.activation))` activation"

    W = layer.weights
    b = layer.bias
    m, n = size(W)
    @inbounds for i in 1:m
        for j in 1:n
            println(io, W[i, j])
        end
        println(io, b[i])
    end
end
