# ========================================
# YAML format
# (load the data with the YAML.jl package)
# ========================================

const ACT_YAML = Dict("Id"=>Id(),
                      "Linear"=>Id(),
                      "ReLU"=>ReLU(),
                      "Sigmoid"=>Sigmoid(),
                      "Tanh"=>Tanh())

"""
    read_nnet_yaml(data::Dict)

Read a neural network from a file in YAML format (see `YAML.jl`) and convert it

### Input

- `data` -- dictionary returned by `YAML.jl` holding the parsed data

### Output

A `FeedforwardNetwork` struct.

### Notes

The following activation functions are supported: identity, relu, sigmoid and tanh;
see `ClosedLoopReachability.ACT_YAML`.
"""
function read_nnet_yaml(data::Dict)
    n_layers = length(data["offsets"])
    layers = []
    for k in 1:n_layers
        weights = data["weights"][k]
        W = copy(reduce(hcat, weights)')
        b = data["offsets"][k]
        a = ACT_YAML[data["activations"][k]]
        L = Layer(W, b, a)
        push!(layers, L)
    end
    return FeedforwardNetwork(layers)
end
