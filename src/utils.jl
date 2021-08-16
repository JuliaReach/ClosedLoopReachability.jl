using NeuralVerification
using NeuralVerification: Layer, n_nodes

_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
_vec(A::AbstractVector) = A

"""
   @modelpath(name)

Return the absolute path to file `name` relative to the executing script.

### Input

- `model_path` -- folder name (ignored by default)
- `name`       -- filename

### Output

A string.

### Notes

This macro is equivalent to `joinpath(@__DIR__, name)`.
The `@modelpath` macro is used in model scripts to load data files relative to the
location of the model, without having to change the directory of the Julia session.
For instance, suppose that the folder `/home/projects/models` contains the script
`my_model.jl`, and suppose that the data file `my_data.dat` located in the same
directory is required to be loaded by `my_model.jl`. Then,

```julia
# suppose the working directory is /home/julia/ and so we ran the script as
# julia -e "include("../projects/models/my_model.jl")"
# in the model file /home/projects/models/my_model.jl we write:
d = open(@modelpath("", "my_data.dat"))
# do stuff with d
```

In this example, the macro `@modelpath("", "my_data.dat")` evaluates to the string
`/home/projects/models/my_data.dat`. If the script `my_model.jl` only had
`d = open("my_data.dat")`, without `@modelpath`, this command would fail as julia
would have looked for `my_data.dat` in the *working* directory, resulting in an
error that the file `/home/julia/my_data.dat` is not found.
"""
macro modelpath(model_path::String, name::String)
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

# ================================================
# Reading a network in Sherlock format
# ================================================

"""
    read_nnet_sherlock(file::String)

Read a neural network from a file in Sherlock format.

### Input

- `file` -- string indicating the location of the input file

### Output

A `Network`.

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
        layers = Vector{Layer}(undef, n_hlayers + 1)

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
            m = layer == 1 ? n_inputs : n_neurons[layer - 1]
            n = layer > n_hlayers ? n_outputs : n_neurons[layer]
            W, b = _read_weights_biases_sherlock(io, m, n)
            # the Sherlock format implicitly uses ReLU activation functions
            layers[layer] = Layer(W, b, ReLU())
        end
    end

    return Network(layers)
end

function _read_weights_biases_sherlock(io, m, n)
    W = Matrix{Float32}(undef, m, n)
    b = Vector{Float32}(undef, n)
    for i in 1:m
        for j in 1:n
            W[i, j] = parse(Float32, readline(io))
        end
        b[i] = parse(Float32, readline(io))
    end
    return W, b
end

"""
    write_nnet_sherlock(nnet::Network, file::String)

Write a neural network to a file in Sherlock format.

### Input

- `nnet` -- neural network
- `file` -- string indicating the location of the output file

### Notes

See [`read_nnet_sherlock`](@ref) for information about the Sherlock format.
"""
function write_nnet_sherlock(nnet::Network, file::String)
    layers = nnet.layers
    n_inputs = size(layers[1].weights, 2)
    n_outputs = n_nodes(layers[end])
    n_hlayers = length(layers) - 1  # includes the output layer
    open(file, "w") do io
        println(io, string(n_inputs))  # number of neurons in input layer
        println(io, string(n_outputs))  # number of neurons in output layer
        println(io, string(n_hlayers))  # number of hidden layers

        # one line for each number of neurons in the hidden layers
        @inbounds for i in 1:n_hlayers
            println(io, string(n_nodes(layers[i])))
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

# ====================
# toy model generation
# ====================

# create a random affine IVP
function random_affine_ivp(n, m)
    A_rand = rand(n, n) .- 0.5  # random values around zero
    b_rand = rand(m) .- 0.5  # random values around zero
    function random_system!(dx, x, params, t; A=A_rand, b=b_rand)
        for i in 1:n
            dx[i] = zero(x[i])
            for j in 1:n
                dx[i] += A[i, j] * x[j]
            end
            for j in 1:m
                dx[i] += b[j] * x[n + j]
            end
        end
        for i in (n+1):(n+m)
            dx[i] = zero(x[i])
        end
    end

    X0 = rand(BallInf, dim=n)
    U0 = ZeroSet(m)  # needed but ignored

    system = BlackBoxContinuousSystem(random_system!, n + m)
    ivp = InitialValueProblem(system, X0 × U0)

    return ivp
end

# create a family of random control problems with the same IVP
function random_control_problems(n::Integer, m::Integer; ivp=nothing, period=0.1)
    if ivp == nothing
        ivp = random_affine_ivp(n, m)
    end

    vars_idx = Dict(:state_vars=>1:n, :control_vars=>(n+1):(n+m))
    problems = ControlledPlant[]

    # create random constant controller
    W = zeros(m, n)
    b = 0.1 .* (rand(m) .- 0.5)  # small random values around zero
    controller = Network([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller
    W = 0.1 .* (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = Network([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller with larger matrix values
    W = 1.1 * (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = Network([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller with large bias values
    W = 0.1 .* (rand(m, n) .- 0.5)  # small random values around zero
    b = 10 * ones(m)
    controller = Network([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random controller with a single layer and ReLU activation function
    W = 1.1 * (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = Network([Layer(W, b, ReLU())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random controller with five layers
    l = 5
    n_max_neurons = 10
    layers = Layer[]
    k_in = n
    for i in 1:l
        k_out = i == l ? m : rand(1:n_max_neurons)
        W = 1.1 * (rand(k_out, k_in) .- 0.5)  # small random values around zero
        b = 0.1 .* (rand(k_out) .- 0.5)  # small random values around zero
        af = i == l ? Id() : ReLU()  # ReLU activation except for the last layer
        push!(layers, Layer(W, b, af))
        k_in = k_out
    end
    controller = Network(layers)
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    return problems
end

output_dim(controller::Network) = size(controller.layers[end].weights, 1)

# relative size between the set-based output and the (CH of) sampled output
function relative_size(X0, nsamples, controller, solver=Ai2())
    @assert output_dim(controller) == 1 "the dimension of the output of the network needs to be 1, but is $output_dim(controller)"
    o = overapproximate(forward_network(solver, controller, X0), Interval)
    u = forward_network(SampledApprox(nsamples), controller, X0)
    return diameter(o)/diameter(u)
end

# print the result `@timed`
function print_timed(stats::NamedTuple)
    Base.time_print(stats.time * 1e9, stats.bytes, stats.gctime * 1e9,
                    Base.gc_alloc_count(stats.gcstats), 0, true)
end
