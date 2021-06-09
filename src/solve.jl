using ReachabilityAnalysis: post
import ReachabilityAnalysis: solve

"""
    solve(prob::AbstractControlProblem, args...; kwargs...)

Solves the controlled problem defined by `prob`

### Input

- `prob`   -- controlled problem

Additional options are passed as arguments or keyword arguments; see the notes
below for details. See the online documentation for examples.

### Output

The solution of a reachability problem controlled by a periodic controller.

### Notes

- Use the `tspan` keyword argument to specify the time span; it can be:
    - a tuple,
    - an interval, or
    - a vector with two components.

- Use the `T` keyword argument to specify the time horizon; the initial time is
  then assumed to be zero.

- Use the `alg_nn` keyword argument to specify the solver for the controller.

- While this function is written with a neural-network controlled systems in
mind, the type of the controller is arbitrary, as long as a function
`forward_network` to analyze it is available.
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

    # get the continuous post or find a default one
    cpost = _get_cpost(ivp, args...; kwargs...)
    if cpost == nothing
        cpost = _default_cpost(ivp, tspan; kwargs...)
    end

    solver = _get_alg_nn(args...; kwargs...)

    splitter = get(kwargs, :splitter, NoSplitter())

    rec_method = get(kwargs, :reconstruction_method,
                     CartesianProductReconstructor())

    remove_zero_generators = get(kwargs, :remove_zero_generators, true)

    sol = _solve(prob, cpost, solver, tvec, τ, splitter, rec_method,
                 remove_zero_generators)

    d = Dict{Symbol, Any}(:solver=>solver)
    return ReachSolution(sol, cpost, d)
end

function _get_alg_nn(args...; kwargs...)
    if haskey(kwargs, :alg_nn)
        solver = kwargs[:alg_nn]
    else
        throw(ArgumentError("the solver for the controller `alg_nn` should be "*
                            "specified, but was not found"))
    end
    return solver
end

# flowpipe with corresponding iteration
struct ControlFlowpipe{FT<:Flowpipe}
    F::FT  # flowpipe
    k::Int  # iteration
end

function _solve(cp::ControlledPlant,
                cpost::AbstractContinuousPost,
                solver::Solver,
                tvec::AbstractVector,
                sampling_time::N,
                splitter::Splitter,
                rec_method::AbstractReconstructionMethod,
                remove_zero_generators::Bool
               ) where {N}

    S = system(cp)
    network = controller(cp)
    st_vars = state_vars(cp)
    in_vars = input_vars(cp)
    ctrl_vars = control_vars(cp)
    normalization = control_normalization(cp)
    preprocessing = control_preprocessing(cp)

    Q₀ = initial_state(cp)
    n = length(st_vars)
    m = length(in_vars)
    q = length(ctrl_vars)
    dim(Q₀) == n + m + q || throw(ArgumentError("dimension mismatch; expect " *
        "the dimension of the initial states of the initial-value problem to " *
        "be $(n + m + q), but it is $(dim(Q₀))"))

    W₀ = m > 0 ? project(Q₀, in_vars) : nothing

    function nnet_forward(X)
        X = apply(preprocessing, X)
        U = forward_network(solver, network, X)
        U = apply(normalization, U)
        if dim(U) == 1  # simplify the control input for intervals
            U = overapproximate(U, Interval)
        end
        return U
    end

    # preallocate output flowpipes
    sol = nothing
    NT = numtype(cpost)
    RT = rsetrep(cpost)
    FT = Flowpipe{NT, RT, Vector{RT}}
    flowpipes = Vector{FT}()
    controls = Vector()

    # waiting list
    waiting_list = Vector{ControlFlowpipe{FT}}()

    # iterate
    first = true
    while true
        if first
            X = nothing
            X₀ = project(Q₀, st_vars)
            k = 1
            first = false
        elseif isempty(waiting_list)
            break
        else
            prev_part = pop!(waiting_list)
            k = prev_part.k + 1
            t = tend(prev_part.F)
            X = prev_part.F(t)
            X₀ = _project_oa(X, st_vars, t;
                             remove_zero_generators=remove_zero_generators) |> set
        end
        t0 = tvec[k]
        t1 = tvec[k+1]
        F, U = _solve_one(X, X₀, W₀, S, st_vars, t0, t1, cpost, nnet_forward,
                            rec_method)
        push!(flowpipes, F)
        push!(controls, U)
        if k < length(tvec) - 1
            push!(waiting_list, ControlFlowpipe(F, k))
        end
    end

    ext = Dict{Symbol, Any}(:controls=>controls)
    return MixedFlowpipe(flowpipes, ext)
end

function _solve_one(X, X₀, W₀, S, st_vars, t0, t1, cpost, nnet_forward, rec_method)
    # add nondeterministic inputs (if any)
    P₀ = isnothing(W₀) ? X₀ : X₀ × W₀

    # get new control inputs from the controller
    U₀ = nnet_forward(X₀)

    # combine states with new control inputs
    Q₀ = _reconstruct(rec_method, P₀, U₀, X, t0)

    dt = t0 .. t1
    sol = post(cpost, IVP(S, Q₀), dt)

    t1′ = tend(sol)
    Δt = t1 - t1′  # difference of exact and actual control time
    @assert LazySets.isapproxzero(Δt) "the flowpipe duration differs " *
        "from the requested duration by $Δt time units (stopped at $(t1′))"
    return sol, U₀
end
