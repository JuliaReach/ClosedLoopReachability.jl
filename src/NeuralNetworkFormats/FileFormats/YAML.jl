"""
    read_YAML(filename::String)

Read a neural network stored in [YAML](https://yaml.org/) format. This function
requires to load the [`YAML.jl` library](https://github.com/JuliaData/YAML.jl).

### Input

- `filename` -- name of the `YAML` file

### Output

A [`FeedforwardNetwork`](@ref).
"""
function read_YAML(filename::String)
    require(@__MODULE__, :YAML; fun_name="read_YAML")

    # read data as a Dict
    data = load_file(filename)

    # read data
    !haskey(data, "weights") && throw(ArgumentError("could not find key `'weights'`"))
    !haskey(data, "offsets") && throw(ArgumentError("could not find key `'offsets'`"))
    !haskey(data, "activations") && throw(ArgumentError("could not find key `'activations'`"))
    weights_vec = data["weights"]
    bias_vec = data["offsets"]
    act_vec = data["activations"]
    n_layer_ops = length(bias_vec)  # number of layer operations

    T = DenseLayerOp{<:ActivationFunction, Matrix{Float64}, Vector{Float64}}
    layers = Vector{T}(undef, n_layer_ops)

    for i in 1:n_layer_ops
        W = weights_vec[i]
        W = Matrix(reduce(hcat, W)')
        b = bias_vec[i]
        act = available_activations[act_vec[i]]
        layers[i] = DenseLayerOp(W, b, act)
    end

    return FeedforwardNetwork(layers)
end
