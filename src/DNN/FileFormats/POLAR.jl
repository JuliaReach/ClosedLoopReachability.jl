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
    function read_activations(io, n_layer_ops)
        activations = [available_activations[readline(io)] for _ in 1:n_layer_ops]
        return i -> activations[i]
    end

    layer_type = DenseLayerOp{<:ActivationFunction, Matrix{Float32}, Vector{Float32}}

    return _read_Sherlock_POLAR(filename, read_activations, layer_type)
end
