# ===============
# Sherlock format
# ===============

"""
    read_nnet_sherlock(file::String)

Read a neural network from a file in Sherlock format.

### Input

- `file` -- string indicating the location of the input file

### Output

A `FeedforwardNetwork`.

### Notes

The Sherlock format is the format used by the tool `Sherlock`.
The format is documented
[here](https://github.com/souradeep-111/sherlock/blob/981fbf8d3bc99f76e0135f2f518e97d2a318cb7c/sherlock-network-format.pdf).
All layers including the output layer implicitly use a ReLU activation function.
"""
function read_nnet_sherlock(file::String)
    layers = nothing
    open(file, "r") do io
        n_inputs = parse(Int, readline(io))  # number of neurons in input layer
        n_outputs = parse(Int, readline(io))  # number of neurons in output layer
        n_hlayers = parse(Int, readline(io))  # number of hidden layers
        T = DenseLayerOp{ReLU, Matrix{Float32}, Vector{Float32}}
        layers = Vector{T}(undef, n_hlayers + 1)

        # one line for each number of neurons in the hidden layers
        n_neurons = Vector{Int}(undef, n_hlayers)
        @inbounds for i in 1:n_hlayers
            n_neurons[i] = parse(Int, readline(io))
        end

        # one line for each weight and bias in the following order:
        # - all incoming weights to neuron 1 in hidden layer 1
        # - bias term of neuron 1 in hidden layer 1
        # - all incoming weights to neuron 2 in hidden layer 1
        # - bias term of neuron 2 in hidden layer 1
        # - ...
        # - all incoming weights to the last neuron in hidden layer 1
        # - bias term of the last neuron in hidden layer 1
        # continue with hidden layer 2 until the output layer
        @inbounds for layer in 1:(n_hlayers + 1)
            m = layer > n_hlayers ? n_outputs : n_neurons[layer]
            n = layer == 1 ? n_inputs : n_neurons[layer - 1]
            W, b = _read_weights_biases_sherlock(io, m, n)
            # the Sherlock format implicitly uses ReLU activation functions
            layers[layer] = DenseLayerOp(W, b, ReLU())
        end
    end

    return FeedforwardNetwork(layers)
end

function _read_weights_biases_sherlock(io, m, n)
    W = Matrix{Float32}(undef, m, n)
    b = Vector{Float32}(undef, m)
    for i in 1:m
        for j in 1:n
            W[i, j] = parse(Float32, readline(io))
        end
        b[i] = parse(Float32, readline(io))
    end
    return W, b
end

"""
    write_nnet_sherlock(nnet::FeedforwardNetwork, file::String)

Write a neural network to a file in Sherlock format.

### Input

- `nnet` -- neural network
- `file` -- string indicating the location of the output file

### Notes

See [`read_nnet_sherlock`](@ref) for information about the Sherlock format.
"""
function write_nnet_sherlock(nnet::FeedforwardNetwork, file::String)
    layers = nnet.layers
    n_inputs = size(layers[1].weights, 2)
    n_outputs = length(layers[end])
    n_hlayers = length(layers) - 1  # includes the output layer
    open(file, "w") do io
        println(io, string(n_inputs))  # number of neurons in input layer
        println(io, string(n_outputs))  # number of neurons in output layer
        println(io, string(n_hlayers))  # number of hidden layers

        # one line for each number of neurons in the hidden layers
        @inbounds for i in 1:n_hlayers
            println(io, string(length(layers[i])))
        end

        # one line for each weight and bias of the hidden and output layers
        @inbounds for layer in layers
            if !(layer.activation isa ReLU)
                @info "the Sherlock format requires ReLU activations, but " *
                    "received a $(typeof(layer.activation)) activation"
            end
            _write_weights_biases_sherlock(io, layer)
        end
    end
    nothing
end

function _write_weights_biases_sherlock(io, layer)
    W = layer.weights
    b = layer.bias
    m, n = size(W)
    for i in 1:m
        for j in 1:n
            println(io, W[i, j])
        end
        println(io, b[i])
    end
end
