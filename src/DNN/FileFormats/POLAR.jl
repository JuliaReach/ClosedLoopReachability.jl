# ============
# POLAR format
# ============

const ACT_POLAR = Dict("Affine"=>Id(),
                       "sigmoid"=>Sigmoid())

function read_nnet_polar(file::String)
    layers = nothing
    open(file, "r") do io
        n_inputs = parse(Int, readline(io))  # number of neurons in input layer
        n_outputs = parse(Int, readline(io))  # number of neurons in output layer
        n_hlayers = parse(Int, readline(io))  # number of hidden layers
        T = DenseLayerOp{<:ActivationFunction, Matrix{Float32}, Vector{Float32}}
        layers = Vector{T}(undef, n_hlayers + 1)

        # one line for each number of neurons in the hidden layers
        n_neurons = Vector{Int}(undef, n_hlayers)
        @inbounds for i in 1:n_hlayers
            n_neurons[i] = parse(Int, readline(io))
        end

        # one line for each activation function
        activations = Vector{ActivationFunction}(undef, n_hlayers + 1)
        @inbounds for i in 1:(n_hlayers + 1)
            activations[i] = ACT_POLAR[readline(io)]
        end

        # the layers use the Sherlock format
        @inbounds for layer in 1:(n_hlayers + 1)
            m = layer > n_hlayers ? n_outputs : n_neurons[layer]
            n = layer == 1 ? n_inputs : n_neurons[layer - 1]
            W, b = _read_weights_biases_sherlock(io, m, n)
            layers[layer] = DenseLayerOp(W, b, activations[layer])
        end
    end

    return FeedforwardNetwork(layers)
end
