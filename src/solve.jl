"""
    solve(prob::AbstractControlProblem, args...; kwargs...)

Solve the control problem defined by `prob`.

### Input

- `prob`   -- controlled problem

Additional options are passed as arguments or keyword arguments; see the notes
below for details. See the online documentation for examples.

### Output

The solution of a reachability problem controlled by a periodic controller.
The control signals are stored in the `ext` field with each flowpipe.

### Notes

- Use the `tspan` keyword argument to specify the time span; it can be:
    - a tuple,
    - an interval, or
    - a vector with two components.

- Use the `T` keyword argument to specify the time horizon; the initial time is
  then assumed to be zero.

- Use the `algorithm_plant` keyword argument to specify the algorithm for the plant.

- Use the `algorithm_controller` keyword argument to specify the algorithm for the controller.

- While this function is written with a neural-network controlled systems in
mind, the type of the controller is arbitrary, as long as a function
`forward` to analyze it is available.
"""
function solve(prob::AbstractControlProblem, args...; kwargs...)
    ivp = plant(prob)

    # check that the dimension of the system and the initial condition match
    _check_dim(ivp)

    τ = period(prob)

    # get vector of time steps
    tspan = _get_tspan(args...; kwargs...)
    T = tend(tspan)
    tvec = range(tstart(tspan), T; step=τ)
    if !(tvec[end] ≈ T)
        tvec = vcat(tvec, T)  # last time interval is shorter
    end

    # get the post operator for the plant, or choose a default one
    algorithm_plant = _get_algorithm_plant(args...; kwargs...)
    if isnothing(algorithm_plant)
        algorithm_plant = _default_cpost(ivp, tspan; kwargs...)
    end

    algorithm_controller = _get_algorithm_controller(args...; kwargs...)

    splitter = get(kwargs, :splitter, NoSplitter())

    input_splitter = get(kwargs, :input_splitter, NoSplitter())

    reconstruction_method = get(kwargs, :reconstruction_method, TaylorModelReconstructor())

    remove_zero_generators = get(kwargs, :remove_zero_generators, true)

    sol = _solve(prob, algorithm_plant, algorithm_controller, tvec, τ, splitter, input_splitter,
                 reconstruction_method, remove_zero_generators)

    d = Dict{Symbol,Any}(:solver_controller => algorithm_controller)
    return ReachSolution(sol, algorithm_plant, d)
end

function _get_algorithm_plant(args...; kwargs...)
    if haskey(kwargs, :algorithm_plant)
        algorithm_controller = kwargs[:algorithm_plant]
    else
        throw(ArgumentError("the algorithm for the plant `algorithm_plant` " *
                            "should be specified, but was not found"))
    end
    return algorithm_controller
end

function _get_algorithm_controller(args...; kwargs...)
    if haskey(kwargs, :algorithm_controller)
        algorithm_controller = kwargs[:algorithm_controller]
    else
        throw(ArgumentError("the algorithm for the controller `algorithm_controller` " *
                            "should be specified, but was not found"))
    end
    return algorithm_controller
end

# element of the waiting list: a flowpipe with corresponding iteration
struct WaitingListElement{FT<:Flowpipe}
    F::FT  # flowpipe
    k::Int  # iteration
end

function _solve(cp::ControlledPlant,
                algorithm_plant::AbstractContinuousPost,
                algorithm_controller::ForwardAlgorithm,
                tvec::AbstractVector,
                sampling_time::N,
                splitter::AbstractSplitter,
                input_splitter::AbstractSplitter,
                reconstruction_method::AbstractReconstructionMethod,
                remove_zero_generators::Bool) where {N}
    S = system(cp)
    controller = ClosedLoopReachability.controller(cp)
    st_vars = states(cp)
    dist_vars = disturbances(cp)
    ctrl_vars = controls(cp)
    preprocessing = control_preprocessing(cp)
    postprocessing = control_postprocessing(cp)

    Q₀ = initial_state(cp)
    n = length(st_vars)
    m = length(dist_vars)
    q = length(ctrl_vars)
    dim(Q₀) == n + m + q || throw(ArgumentError("dimension mismatch; " *
                                                "expected the dimension of the initial states of the initial-value " *
                                                "problem to be $(n + m + q), but it is $(dim(Q₀))"))

    W₀ = m > 0 ? project(Q₀, dist_vars) : nothing

    # preallocate output flowpipes
    sol = nothing
    NT = numtype(algorithm_plant)
    RT = rsetrep(algorithm_plant)
    FT = Flowpipe{NT,RT,Vector{RT}}
    flowpipes = Vector{FT}()

    # waiting list
    waiting_list = Vector{WaitingListElement{FT}}()

    # first step
    k = 1
    R = nothing
    t0 = tvec[k]
    t1 = tvec[k + 1]
    X₀ = project(Q₀, st_vars)
    X₀s = haskey(splitter, k) ? split(splitter[k], X₀) : [X₀]
    results = Vector{Vector{Flowpipe}}(undef, length(X₀s))

    # first perform an isolated analysis because of problems in TaylorSeries
    # (global variables need to be written once)
    @inbounds results[1] = _solve_one(R, first(X₀s), W₀, S, st_vars, t0, t1, algorithm_plant,
                                      reconstruction_method, algorithm_controller, controller,
                                      preprocessing, postprocessing, input_splitter)
    # parallelize analysis of the remaining parts
    Threads.@threads for i in 2:length(results)
        @inbounds results[i] = _solve_one(R, X₀s[i], W₀, S, st_vars, t0, t1, algorithm_plant,
                                          reconstruction_method, algorithm_controller, controller,
                                          preprocessing, postprocessing, input_splitter)
    end
    # collect results from all threads
    for Fs in results
        append!(flowpipes, Fs)
        if k < length(tvec) - 1
            for F in Fs
                push!(waiting_list, WaitingListElement(F, k))
            end
        end
    end

    # iteration
    while !isempty(waiting_list)
        prev_part = pop!(waiting_list)
        k = prev_part.k + 1
        t = tend(prev_part.F)
        R = prev_part.F(t)
        X₀ = set(_project_oa(R, st_vars, t;
                             remove_zero_generators=remove_zero_generators))
        t0 = tvec[k]
        t1 = tvec[k + 1]
        X₀s = haskey(splitter, k) ? split(splitter[k], X₀) : [X₀]
        results = Vector{Vector{Flowpipe}}(undef, length(X₀s))
        # parallelize analysis
        Threads.@threads for i in 1:length(results)
            @inbounds results[i] = _solve_one(R, X₀s[i], W₀, S, st_vars, t0, t1, algorithm_plant,
                                              reconstruction_method, algorithm_controller,
                                              controller, preprocessing, postprocessing,
                                              input_splitter)
        end
        # collect results from all threads
        for Fs in results
            append!(flowpipes, Fs)
            if k < length(tvec) - 1
                for F in Fs
                    push!(waiting_list, WaitingListElement(F, k))
                end
            end
        end
    end

    return MixedFlowpipe(flowpipes)
end

function controller_forward(algorithm_controller, controller, X, preprocessing, postprocessing)
    X = apply(preprocessing, X)
    U = forward(X, controller, algorithm_controller)
    U = apply(postprocessing, U)
    if dim(U) == 1  # simplify the control input for intervals
        U = overapproximate(U, Interval)
    end
    return U
end

function _solve_one(R, X₀, W₀, S, st_vars, t0, t1, algorithm_plant, reconstruction_method,
                    algorithm_controller, controller, preprocessing, postprocessing, splitter)
    # add disturbances (if any)
    P₀ = isnothing(W₀) ? X₀ : X₀ × W₀

    # get new control inputs from the controller
    U = controller_forward(algorithm_controller, controller, X₀, preprocessing, postprocessing)

    dt = t0 .. t1

    # split control inputs
    sols = Flowpipe[]
    for Ui in split(splitter, U)
        # combine states with new control inputs
        Q₀ = _reconstruct(reconstruction_method, P₀, Ui, R, t0)

        sol = post(algorithm_plant, IVP(S, Q₀), dt)

        t1′ = tend(sol)
        Δt = t1 - t1′  # difference of exact and actual control time
        @assert isapproxzero(Δt) "the flowpipe duration differs from the " *
                                 "requested duration by $Δt time units (stopped at $(t1′))"
        push!(sols, sol)

        # attach control signals to flowpipe
        sol.ext[:controls] = Ui
    end

    return sols
end
