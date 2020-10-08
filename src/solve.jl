using ReachabilityAnalysis: post

abstract type AbstractNeuralNetworkControlProblem end

# ST: type of system
# XT: type of initial condition
struct ControlledPlant{ST, XT, IV<:InitialValueProblem{ST, XT}, DT} <: AbstractNeuralNetworkControlProblem
    ivp::IV
    controller::Network
    vars::Dict{Symbol, DT}
end

plant(cp::ControlledPlant) = cp.ivp
system(cp::ControlledPlant) = cp.ivp.s
controller(cp::ControlledPlant) = cp.controller

function state_vars(cp::ControlledPlant)
    try
        cp.vars[:state_vars]
    catch error
        isa(error, KeyError) && println("key `:state_vars` not found")
    end
end

function input_vars(cp::ControlledPlant)
    try
        cp.vars[:input_vars]
    catch error
        isa(error, KeyError) && println("key `:input_vars` not found")
    end
end

function control_vars(cp::ControlledPlant)
    try
        cp.vars[:control_vars]
    catch error
        isa(error, KeyError) && println("key `:control_vars` not found")
    end
end

#=
NOTES:

- Add default neural network solver.

=#
"""
    solve(prob::AbstractNeuralNetworkControlProblem, args...; kwargs...)

Solves the neural network controlled problem defined by `prob`

### Input

- `prob`   -- neural network controlled problem

Additional options are passed as arguments or keyword arguments; see the notes
below for details. See the online documentation for examples.

### Output

The solution of a reachability problem controlled by a neural network.

### Notes

- Use the `tspan` keyword argument to specify the time span; it can be:
    - a tuple,
    - an interval, or
    - a vector with two components.

- Use the `T` keyword argument to specify the time horizon; the initial time is
  then assumed to be zero.

- Use the `Tsample` or `sampling_time` keyword argument to specify the sampling
time for the model.

- Use the `alg_nn` keyword argument to specify the the neural network solver.

"""
function solve(prob::AbstractNeuralNetworkControlProblem, args...; kwargs...)
    ivp = plant(prob)

    # check that the dimension of the system and the initial condition match
    _check_dim(ivp)

    # get time span
    tspan = _get_tspan(args...; kwargs...)

    # get the continuous post or find a default one
    cpost = _get_cpost(ivp, args...; kwargs...)
    if cpost == nothing
        cpost = _default_cpost(ivp, tspan; kwargs...)
    end

    # extract the sampling time
    Tsample = _get_Tsample(; kwargs...)

    solver = _get_alg_nn(args...; kwargs...)

    if haskey(kwargs, :apply_initial_control)
        init_ctrl = kwargs[:apply_initial_control]
    else
        init_ctrl = true
    end

    if haskey(kwargs, :preprocess)
        preprocess = kwargs[:preprocess]
    else
        preprocess = X -> overapproximate(X, Hyperrectangle)
    end

    sol = _solve(prob, cpost, solver, tspan, Tsample, init_ctrl, preprocess)

    d = Dict{Symbol, Any}(:solver=>solver)
    return ReachSolution(sol, cpost, d)
end

function _get_Tsample(; kwargs...)
    if haskey(kwargs, :sampling_time)
        Tsample = kwargs[:sampling_time]
    elseif haskey(kwargs, :Tsample)
        Tsample = kwargs[:Tsample]
    else
        throw(ArgumentError("the sampling time `Tsample` should be specified, but was not found"))
    end
    return Tsample
end

function _get_alg_nn(args...; kwargs...)
    if haskey(kwargs, :alg_nn)
        solver = kwargs[:alg_nn]
    else
        throw(ArgumentError("the solver for the neural network `alg_nn` should be specified, but was not found"))
    end
    return solver
end

function _solve(cp::ControlledPlant,
                cpost::AbstractContinuousPost,
                solver,#::Solver,
                time_span::TimeInterval,
                sampling_time::N,
                apply_initial_control::Bool,
                preprocess::Function # function that is applied before passing the set to the neural network controller
                ) where {N}

    ivp = plant(cp)
    S = system(cp)
    network = controller(cp)
    st_vars = state_vars(cp)
    in_vars = input_vars(cp)
    ctrl_vars = control_vars(cp)
    controls = Dict()

    Q₀ = initial_state(ivp) # TODO initial_state(plant)
    n = length(st_vars)
    m = length(in_vars)
    q = length(ctrl_vars)
    dim(Q₀) == n + m + q || throw(ArgumentError("dimension mismatch; expect the dimension of the initial states " *
         "of the initial-value problem to be $(n + m + q), but it is $(dim(Q₀))"))

    X₀ = LazySets.Projection(Q₀, st_vars)

    if !isempty(in_vars)
        W₀ = LazySets.Projection(Q₀, in_vars)
        P₀ = X₀ × W₀
    else
        P₀ = X₀
    end

    if apply_initial_control
        X0aux = preprocess(X₀)
        if solver == "hybrid"
            U₀ = forward(network, X0aux)
        else
            U₀ = forward_network(solver, network, X0aux)
        end
    else
        U₀ = LazySets.Projection(Q₀, ctrl_vars)
    end

    ti = tstart(time_span)
    NSAMPLES = ceil(Int, diam(time_span) / sampling_time)

    # preallocate output flowpipe
    NT = numtype(cpost)
    RT = rsetrep(cpost)
    FT = Flowpipe{NT, RT, Vector{RT}}
    out = Vector{FT}(undef, NSAMPLES)

    for i = 1:NSAMPLES
        if isa(U₀, LazySet)
            Q₀ = P₀ × U₀
        else
            # TODO should take eg. convex hull if the network retuns > 1 set
            Q₀ = P₀ × first(U₀)
        end

        controls[i] = U₀
        dt = ti .. (ti + sampling_time)
        sol = post(cpost, IVP(S, Q₀), dt)
        out[i] = sol

        ti += sampling_time

        X = sol[end] # reach-set
        X₀ = _Projection(X, st_vars) |> set # lazy set
        P₀ = isempty(in_vars) ? X₀ : X₀ × W₀

        if solver == "hybrid"
            U₀ = forward(network, preprocess(X₀))
        else
            U₀ = forward_network(solver, network, preprocess(X₀))
        end
    end

    return MixedFlowpipe(out, Dict{Symbol,Any}(:controls=>controls))
end
