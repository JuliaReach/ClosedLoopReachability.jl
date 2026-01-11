# ====================
# toy model generation
# ====================

# create a random affine IVP
function random_affine_ivp(n, m)
    A_rand = rand(n, n) .- 0.5  # random values around zero
    b_rand = rand(m) .- 0.5  # random values around zero
    function random_system!(dx, x, p, t; A=A_rand, b=b_rand)
        for i in 1:n
            dx[i] = zero(x[i])
            for j in 1:n
                dx[i] += A[i, j] * x[j]
            end
            for j in 1:m
                dx[i] += b[j] * x[n + j]
            end
        end
        for i in (n + 1):(n + m)
            dx[i] = zero(x[i])
        end
    end

    X0 = rand(BallInf; dim=n)
    U0 = ZeroSet(m)  # needed but ignored

    system = BlackBoxContinuousSystem(random_system!, n + m)
    ivp = InitialValueProblem(system, X0 × U0)

    return ivp
end

# create a family of random control problems with the same IVP
function random_control_problems(n::Integer, m::Integer; ivp=nothing, period=0.1)
    if isnothing(ivp)
        ivp = random_affine_ivp(n, m)
    end

    vars_idx = Dict(:states => 1:n, :controls => (n + 1):(n + m))
    problems = ControlledPlant[]

    # create random constant controller
    W = zeros(m, n)
    b = 0.1 .* (rand(m) .- 0.5)  # small random values around zero
    controller = FeedforwardNetwork([DenseLayerOp(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller
    W = 0.1 .* (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = FeedforwardNetwork([DenseLayerOp(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller with larger matrix values
    W = 1.1 * (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = FeedforwardNetwork([DenseLayerOp(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random affine controller with large bias values
    W = 0.1 .* (rand(m, n) .- 0.5)  # small random values around zero
    b = 10 * ones(m)
    controller = FeedforwardNetwork([DenseLayerOp(W, b, Id())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random controller with a single layer and ReLU activation function
    W = 1.1 * (rand(m, n) .- 0.5)  # small random values around zero
    b = zeros(m)
    controller = FeedforwardNetwork([DenseLayerOp(W, b, ReLU())])
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    # create random controller with five layers
    l = 5
    n_max_neurons = 10
    T = DenseLayerOp{<:ActivationFunction,Matrix{Float64},Vector{Float64}}
    layers = T[]
    k_in = n
    for i in 1:l
        k_out = i == l ? m : rand(1:n_max_neurons)
        W = 1.1 * (rand(k_out, k_in) .- 0.5)  # small random values around zero
        b = 0.1 .* (rand(k_out) .- 0.5)  # small random values around zero
        af = i == l ? Id() : ReLU()  # ReLU activation except for the last layer
        push!(layers, DenseLayerOp(W, b, af))
        k_in = k_out
    end
    controller = FeedforwardNetwork(layers)
    push!(problems, ControlledPlant(ivp, controller, vars_idx, period))

    return problems
end
