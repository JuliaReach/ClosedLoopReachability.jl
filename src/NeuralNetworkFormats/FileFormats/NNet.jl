"""
    read_NNet(filename::String)

Read a neural network stored in [`NNet`](https://github.com/sisl/NNet) format.

### Input

- `filename` -- name of the `NNet` file

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

The format assumes that all layers but the output layer use ReLU activation (the
output layer uses the identity activation).

The format looks like this (each line may optionally be terminated by a comma):

1. Header text, each line beginning with "//"
2. Comma-separated line with four values:
   number of layer operations, number of inputs, number of outputs, maximum
   layer size
3. Comma-separated line with the layer sizes
4. Flag that is no longer used
5. Minimum values of inputs
6. Maximum values of inputs
7. Mean values of inputs and one value for all outputs
8. Range values of inputs and one value for all outputs
9. Blocks of lines describing the weight matrix and bias vector for a layer;
   each matrix row is written as a comma-separated line, and each vector entry
   is written in its own line

The code follows [this implementation](https://github.com/sisl/NeuralVerification.jl/blob/957cb32081f37de57d84d7f0813f708288b56271/src/utils/util.jl#L10).
"""
function read_NNet(filename::String)
    layers = nothing
    open(filename, "r") do io
        line = readline(io)

        # skip header text
        while startswith(line, "//")
            line = readline(io)
        end

        # four numbers: only the first (number of layer operations) is relevant
        n_layer_ops = parse(Int, split(line, ",")[1])

        # layer sizes
        layer_sizes = parse.(Int, split(readline(io), ",")[1:n_layer_ops+1])

        # five lines of irrelevant information
        for i in 1:5
            line = readline(io)
        end

        # read layers except for the output layer (with ReLU activation)
        T = DenseLayerOp{<:ActivationFunction, Matrix{Float64}, Vector{Float64}}
        layers = T[_read_layer_NNet(dim, io, ReLU()) for dim in layer_sizes[2:end-1]]

        # read output layer (with identity activation)
        push!(layers, _read_layer_NNet(last(layer_sizes), io, Id()))
    end

    return FeedforwardNetwork(layers)
end

# some complication because lines can optionally be terminated by a comma
function _read_layer_NNet(output_dim::Int, f::IOStream, act)
     # simple parsing as a Vector of Vectors
     weights = [parse.(Float64, filter(!isempty, split(readline(f), ","))) for _ in 1:output_dim]
     weights = vcat(weights'...)
     bias = [parse(Float64, split(readline(f), ",")[1]) for _ in 1:output_dim]
     return DenseLayerOp(weights, bias, act)
end
