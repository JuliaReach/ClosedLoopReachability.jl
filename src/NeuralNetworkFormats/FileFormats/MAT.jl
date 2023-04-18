"""
    read_MAT(filename::String; act_key::String)

Read a neural network stored in MATLAB's
[`MAT`](https://www.mathworks.com/help/matlab/import_export/load-parts-of-variables-from-mat-files.html)
format. This function requires to load the
[`MAT.jl` library](https://github.com/JuliaIO/MAT.jl).

### Input

- `filename` -- name of the `MAT` file
- `act_key`  -- key used for the activation functions

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

The `MATLAB` file encodes a dictionary, which is assumed to contain the
following:

- A vector of weight matrices (under the name `"W"`)
- A vector of bias vectors (under the name `"b"`)
- A vector of strings for the activation functions (under the name passed via
  `act_key`)
"""
function read_MAT(filename::String; act_key::String)
    require(@__MODULE__, :MAT; fun_name="read_MAT")

    # read data as a Dict
    data = matread(filename)

    # read data
    !haskey(data, "W") && throw(ArgumentError("could not find key `'W'`"))
    !haskey(data, "b") && throw(ArgumentError("could not find key `'b'`"))
    weights_vec = data["W"]
    bias_vec = data["b"]
    act_vec = data[act_key]
    n_layer_ops = length(bias_vec)  # number of layer operations

    T = DenseLayerOp{<:ActivationFunction, Matrix{Float64}, Vector{Float64}}
    layers = Vector{T}(undef, n_layer_ops)

    for i in 1:n_layer_ops
        # weights
        W = weights_vec[i]
        s = size(W)
        if length(s) == 4
            # weights sometimes are stored as a multi-dimensional array
            # with two flat dimensions
            if s[3] == 1 && s[4] == 1
                W = reshape(W, s[1], s[2])
            else
                throw(ArgumentError("unexpected dimension of the weights matrix: $s"))
            end
        end

        # bias
        b = bias_vec[i]

        # activation function
        act = available_activations[act_vec[i]]

        layers[i] = DenseLayerOp(W, _vec(b), act)
    end

    return FeedforwardNetwork(layers)
end

# convert to a Vector
_vec(A::Vector) = A
_vec(A::AbstractVector) = Vector(A)
_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
