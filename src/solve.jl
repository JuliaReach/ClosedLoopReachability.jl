using ReachabilityAnalysis: post
import ReachabilityAnalysis: solve

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

    τ = period(prob)

    solver = _get_alg_nn(args...; kwargs...)

    init_ctrl = get(kwargs, :apply_initial_control, true)

    splitter = get(kwargs, :splitter, NoSplitter())

    rec_method = get(kwargs, :reconstruction_method, CartesianProductReconstructor())

    sol = _solve(prob, cpost, solver, tspan, τ, init_ctrl, splitter, rec_method)

    d = Dict{Symbol, Any}(:solver=>solver)
    return ReachSolution(sol, cpost, d)
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
                solver::Solver,
                time_span::TimeInterval,
                sampling_time::N,
                apply_initial_control::Bool,
                splitter::Splitter,
                rec_method::AbstractReconstructionMethod
               ) where {N}

    S = system(cp)
    network = controller(cp)
    st_vars = state_vars(cp)
    in_vars = input_vars(cp)
    ctrl_vars = control_vars(cp)
    controls = Dict()
    normalization = control_normalization(cp)
    preprocessing = control_preprocessing(cp)

    Q₀ = initial_state(cp)
    n = length(st_vars)
    m = length(in_vars)
    q = length(ctrl_vars)
    dim(Q₀) == n + m + q || throw(ArgumentError("dimension mismatch; expect the dimension of the initial states " *
         "of the initial-value problem to be $(n + m + q), but it is $(dim(Q₀))"))

    X₀ = project(Q₀, st_vars)

    if !isempty(in_vars)
        W₀ = project(Q₀, in_vars)
        P₀ = X₀ × W₀
    else
        P₀ = X₀
    end

    if apply_initial_control
        Us = Vector{splitter.output_type}()
        for X₀ in split(splitter, X₀)
            X0aux = apply(preprocessing, X₀)
            U₀ = forward_network(solver, network, X0aux)
            U₀ = apply(normalization, U₀)
            push!(Us, U₀)
        end
        U₀ = merge(splitter, UnionSetArray(Us))
    else
        U₀ = project(Q₀, ctrl_vars)
    end

    ti = tstart(time_span)
    NSAMPLES = ceil(Int, diam(time_span) / sampling_time)

    # preallocate output flowpipe
    NT = numtype(cpost)
    RT = rsetrep(cpost)
    FT = Flowpipe{NT, RT, Vector{RT}}
    out = Vector{FT}(undef, NSAMPLES)

    X = nothing

    for i = 1:NSAMPLES

        # simplify the control input for intervals
        if dim(U₀) == 1
            U₀ = overapproximate(U₀, Interval)
        end

        Q₀ = _reconstruct(rec_method, P₀, U₀, X, ti)

        Ti = i < NSAMPLES ? (ti + sampling_time) : tend(time_span)

        controls[i] = U₀
        dt = ti .. Ti
        sol = post(cpost, IVP(S, Q₀), dt)
        out[i] = sol

        if i == NSAMPLES
            # no need to ask the network again
            break
        end

        ti = min(Ti, tend(sol))
        @assert LazySets._isapprox(Ti, tend(sol))

        X = sol(ti)
        X₀ = _project_oa(X, st_vars, ti) |> set
        P₀ = isempty(in_vars) ? X₀ : X₀ × W₀

        Us = Vector{splitter.output_type}()
        for X₀ in split(splitter, X₀)
            X0aux = apply(preprocessing, X₀)
            U₀ = forward_network(solver, network, X0aux)
            U₀ = apply(normalization, U₀)
            push!(Us, U₀)
        end
        U₀ = merge(splitter, UnionSetArray(Us))
    end

    ext = Dict{Symbol, Any}(:controls=>controls)
    return MixedFlowpipe(out, ext)
end
