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

    vars_idx = Dict(:states=>1:n, :controls=>(n+1):(n+m))
    problems = ControlledPlant[]

    # create random constant controller
    W = zeros(m, n)
    b = 0.1 .* (rand(m) .- 0.5)  # small random values around zero
    controller = FeedforwardNetwork([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller
    W = 0.1 .* (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = FeedforwardNetwork([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller with larger matrix values
    W = 1.1 * (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = FeedforwardNetwork([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller with large bias values
    W = 0.1 .* (rand(m, n) .- 0.5)  # small random values around zero
    b = 10 * ones(m)
    controller = FeedforwardNetwork([Layer(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random controller with a single layer and ReLU activation function
    W = 1.1 * (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = FeedforwardNetwork([Layer(W, b, ReLU())])
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
    controller = FeedforwardNetwork(layers)
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    return problems
end

output_dim(controller::FeedforwardNetwork) = size(controller.layers[end].weights, 1)

# relative size between the set-based output and the (CH of) sampled output
function relative_size(X0, nsamples, controller, solver=DeepZ())
    @assert output_dim(controller) == 1 "the dimension of the output of the network needs to be 1, but is $output_dim(controller)"
    o = overapproximate(forward(solver, controller, X0), Interval)
    u = forward(SampledApprox(nsamples), controller, X0)
    return diameter(o)/diameter(u)
end
