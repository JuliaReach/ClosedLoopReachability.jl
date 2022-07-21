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
# Reading a network in NNET format
# ================================================

"""
    read_nnet(file; final_activation=Id())

Read a neural network stored in a `.nnet` file.

### Input

- `file`             -- string indicating the location of the `.nnet` file
- `final_activation` -- (optional, default: `Id()`) activation function of the
                        last layer

### Output

A `Network` struct.

### Notes

For more info on the `.nnet` format, see [here](https://github.com/sisl/NNet).
The format assumes all hidden layers have ReLU activation except for the last
one.

The code is taken from
[here](https://github.com/sisl/NeuralVerification.jl/blob/237f0924eaf1988188219aff4360a677534f3871/src/utils/util.jl#L10).
"""
function read_nnet(file::String; final_activation=Id())
    f = open(file)
    line = readline(f)
    while occursin("//", line)  # skip comments
        line = readline(f)
    end
    nlayers = parse(Int64, split(line, ",")[1])
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    for i in 1:5
        line = readline(f)
    end
    layers = Layer[_read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, _read_layer(last(layer_sizes), f, final_activation))
    return Network(layers)
end

function _read_layer(output_dim::Int64, f::IOStream, act = ReLU())
     rowparse(splitrow) = parse.(Float64, splitrow[findall(!isempty, splitrow)])
     W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
     weights = vcat(W_str_vec'...)
     bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
     bias = rowparse(bias_string)
     return Layer(weights, bias, act)
end

# ================================================
# Reading a network in MAT format
# ================================================

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

A `Network` struct.

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
    return Network(layers)
end

# ================================================
# Reading a network in ONNX format
# (load the data with ONNX.jl)
# ================================================

const ACT_ONNX = Dict("σ"=>Sigmoid())

"""
    read_nnet_onnx(data::Dict)

Read a neural network from a file in ONNX format (see `ONNX.jl`) and convert it.

### Input

- `data` -- an `ONNXCtx` struct parsed by `ONNX.jl`

### Output

A `Network` struct.

### Notes

This implementation does not apply to general `ONNX` networks because it assumes
a specific structure:
1. First comes a bias vector for the input vector that is a zero vector.
2. Next come the weight matrices `W` (transposed) and bias vectors `b` in pairs
*in the order in which they are applied*.
3. Next come the affine maps and the activation functions *in the order in which
they are applied*. The last layer does not have an activation function.

Some of these assumptions *are not checked*. Hence it may happen that it returns
a result that is incorrect. A general implementation may be added in the future.

The following activation function is supported: sigmoid (and an implicit
identity in the last layer); see `ClosedLoopReachability.ACT_ONNX`.
"""
function read_nnet_onnx(data)
    if !isdefined(@__MODULE__, :ONNX)
        error("package `ONNX` is required for `read_nnet_onnx`")
    end

    @assert data isa Ghost.Tape{ONNX.ONNXCtx} "`read_nnet_onnx` must be " *
        "called with `ONNX.Ghost.Tape{ONNX.ONNXCtx}`"

    layer_parameters = []
    ops = data.ops
    @assert ops[1] isa Ghost.Input && iszero(ops[1].val)  # skip input operation
    idx = 2
    @inbounds while idx <= length(ops)
        op = ops[idx]
        if !(op.val isa AbstractMatrix)
            break
        end
        W = permutedims(op.val)
        idx += 1
        op = ops[idx]
        @assert op.val isa AbstractVector "expected a bias vector"
        b = op.val
        push!(layer_parameters, (W, b))
        idx += 1
    end
    n_layers = div(idx - 2, 2)
    @assert length(ops) == 4 * n_layers
    layers = Layer[]
    layer = 1
    while idx <= length(ops)
        # affine map (treated implicitly)
        op = ops[idx]
        @assert op isa Ghost.Call "expected an affine map"
        args = op.args
        @assert length(args) == 5
        @assert args[2] == onnx_gemm
        @assert args[3]._op.id == (layer == 1 ? 1 : idx - 1)
        @assert args[4]._op.id == 2 * layer
        @assert args[5]._op.id == 2 * layer + 1
        W, b = @inbounds layer_parameters[layer]
        idx += 1

        # activation function
        if idx > length(ops)
            # last layer is assumed to be the identity
            a = Id()
        else
            op = ops[idx]
            @assert op isa Ghost.Call "expected an activation function"
            args = op.args
            @assert length(args) == 2
            @assert args[2]._op.id == idx - 1
            a = ACT_ONNX[string(args[1])]
            idx += 1
        end

        L = Layer(W, b, a)
        push!(layers, L)
        layer += 1
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
            m = layer > n_hlayers ? n_outputs : n_neurons[layer]
            n = layer == 1 ? n_inputs : n_neurons[layer - 1]
            W, b = _read_weights_biases_sherlock(io, m, n)
            # the Sherlock format implicitly uses ReLU activation functions
            layers[layer] = Layer(W, b, ReLU())
        end
    end

    return Network(layers)
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

# ================================================
# Reading a network in POLAR format
# ================================================

const ACT_POLAR = Dict("Affine"=>Id(),
                       "sigmoid"=>Sigmoid())

function read_nnet_polar(file::String)
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
            layers[layer] = Layer(W, b, activations[layer])
        end
    end

    return Network(layers)
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
function relative_size(X0, nsamples, controller, solver=DeepZ())
    @assert output_dim(controller) == 1 "the dimension of the output of the network needs to be 1, but is $output_dim(controller)"
    o = overapproximate(forward(solver, controller, X0), Interval)
    u = forward(SampledApprox(nsamples), controller, X0)
    return diameter(o)/diameter(u)
end

# print the result `@timed`
function print_timed(stats::NamedTuple)
    Base.time_print(stats.time * 1e9, stats.bytes, stats.gctime * 1e9,
                    Base.gc_alloc_count(stats.gcstats), 0, true)
end
