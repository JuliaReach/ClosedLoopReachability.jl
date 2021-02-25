using NeuralVerification, MAT
using NeuralVerification: Layer, Network, ReLU, Id

_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
_vec(A::AbstractVector) = A

"""
   @relpath(name)

Return the absolute path to file `name` relative to the executing script.

### Input

- `name` -- filename

### Output

A string.

### Notes

This macro is equivalent to `joinpath(@__DIR__, name)`.
The `@relpath` macro is used in model scripts to load data files relative to the
location of the model, without having to change the directory of the Julia session.
For instance, suppose that the folder `/home/projects/models` contains the script
`my_model.jl`, and suppose that the data file `my_data.dat` located in the same
directory is required to be loaded by `my_model.jl`. Then,

```julia
# suppose the working directory is /home/julia/ and so we ran the script as
# julia -e "include("../projects/models/my_model.jl")"
# in the model file /home/projects/models/my_model.jl we write:
d = open(@relpath "my_data.dat")
# do stuff with d
```

In this example, the macro `@relpath "my_data.dat"` evaluates to the string
`/home/projects/models/my_data.dat`. If the script `my_model.jl` only had
`d = open("my_data.dat")`, without `@relpath`, this command would fail as julia
would have looked for `my_data.dat` in the *working* directory, resulting in an
error that the file `/home/julia/my_data.dat` is not found.
"""
macro relpath(name::String)
    __source__.file === nothing && return nothing
    _dirname = dirname(String(__source__.file))
    dir = isempty(_dirname) ? pwd() : abspath(_dirname)
    return joinpath(dir, name)
end

# ================================================
# Reading a network in MAT format
# ================================================

"""
    read_nnet_mat(file; key=nothing, act_key="activation_fcns")

Read a neural network stored in a `.mat` file and return the corresponding network
in the format of `NeuralVerification.jl`.

### Input

- `file`    -- string indicating the location of the `.mat` file containing the neural network
- `key`     -- (optional, default: `nothing`) key used to search the dictionary containing the controller;
               by default we search the top-level dictionary; a typical value is `"controller"`
- `act_key` -- (optional, default: `"activation_fcns"`) key used to search the activation
               functions; typical values are `"activation_fcns"` or `"act_fcns"`

### Output

A `Network` struct.

### Notes

The following activation functions are supported:

- RELU: "relu" (`ReLU`)
- Identity: "linear" (`Id`)
"""
function read_nnet_mat(file::String; key=nothing, act_key="activation_fcns")
    isdefined(@__MODULE__, :MAT) || error("package 'MAT' is required")
    vars = matread(file)

    # some models store the controller under a specified key
    if !isnothing(key)
        !haskey(vars, key) && throw(ArgumentError("didn't find key $key, existing keys are $(keys(vars))"))
        dic = vars[key]
    else
        dic = vars
    end

    # get number of layers either from a dictionary entry or from the length of the weights array
    if haskey(dic, "number_of_layers")
        m = Int(dic["number_of_layers"])
    else
        m = length(dic["W"])
    end
    layers = Vector{Layer}(undef, m)
    aF = dic[act_key]

    for n = 1:m
        W = dic["W"][n]
        b = dic["b"][n]
        if aF[n] == "relu"
            act = ReLU()
        elseif aF[n] == "linear"
            act = Id()
        else
            error("error, aF = $(aF[n]), nLayer = $n")
        end
        layers[n] = Layer(W, _vec(b), act)
    end

    return Network(layers)
end

# ================================================
# Reading a network in YAML format
# (load the data with YAML.jl)
# ================================================

const ACT_YAML = Dict("Id"=>Id(),
                      "ReLU"=>ReLU(),
                      "Sigmoid"=>Sigmoid(),
                      "Tanh"=>Tanh())

"""
    read_nnet_yaml(data::Dict)

Read a neural network from a file in YAML format (see `YAML.jl`) and convert it

Read a neural network stored in a `.mat` file and return the corresponding network
in the format of `NeuralVerification.jl`.

### Input

- `file` -- string indicating the location of the `.mat` file containing the neural network
- `key`  -- (optional, default: `"controller"`) key used to search the dictionary

### Output

A `Network` struct.

### Notes

The following activation functions are supported: identity, relu, sigmoid and tanh;
see `NeuralNetworkAnalysis.ACT_YAML`.
"""
function read_nnet_yaml(data::Dict)
    NLAYERS = length(data["offsets"])
    layers = []
    for k in 1:NLAYERS
        weights = data["weights"][k]
        W = copy(reduce(hcat, weights)')
        b = data["offsets"][k]
        a = ACT_YAML[data["activations"][k]]
        L = Layer(W, b, a)
        push!(layers, L)
    end
    return Network(layers)
end
