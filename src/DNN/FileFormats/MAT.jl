# ==========
# MAT format
# ==========

const ACT_MAT = Dict("linear"=>Id(),
                     "relu"=>ReLU(),
                     "sigmoid"=>Sigmoid(),
                     "tanh"=>Tanh())

"""
    read_nnet_mat(file; key=nothing, act_key="activation_fcns")

Read a neural network stored in a `.mat` file.

### Input

- `file`    -- string indicating the location of the `.mat` file containing the neural network
- `key`     -- (optional, default: `nothing`) key used to search the dictionary containing the controller;
               by default we search the top-level dictionary; a typical value is `"controller"`
- `act_key` -- (optional, default: `"activation_fcns"`) key used to search the activation
               functions; typical values are `"activation_fcns"` or `"act_fcns"`

### Output

A `FeedforwardNetwork` struct.

### Notes

The following activation functions are supported: identity, relu, sigmoid, and
tanh; see `ClosedLoopReachability.ACT_MAT`.
"""
function read_nnet_mat(file::String; key=nothing, act_key="activation_fcns")
    require(@__MODULE__, :MAT; fun_name="read_nnet_mat")

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
    T = DenseLayerOp{<:ActivationFunction, Matrix{Float64}, Vector{Float64}}
    layers = Vector{T}(undef, m)
    aF = dic[act_key]

    for n = 1:m

        # weights matrix
        W = dic["W"][n]
        s = size(W)
        if length(s) == 4
            # models sometimes are stored as a multi-dim array
            # with two dimensions which are flat
            if s[3] == 1 && s[4] == 1
                W = reshape(W, s[1], s[2])
            else
                throw(ArgumentError("unexpected dimension of the weights matrix: $s"))
            end
        end

        # bias
        b = dic["b"][n]

        # activation function
        act = ACT_MAT[aF[n]]

        layers[n] = DenseLayerOp(W, _vec(b), act)
    end

    return FeedforwardNetwork(layers)
end

_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
_vec(A::AbstractVector) = A
