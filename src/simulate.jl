struct SimulationSolution{TT,CT,IT}
    trajectory::TT  # trajectory pieces for each control cycle
    controls::CT  # control inputs for each control cycle
    disturbances::IT  # disturbances for each control cycle
end

struct EnsembleSimulationSolution{TT,CT,IT}
    solutions::Vector{SimulationSolution{TT,CT,IT}}
end

"""
    simulate(cp::AbstractControlProblem, args...; kwargs...)

Simulate a controlled system for a family of random trajectories.

### Input

- `cp`           -- controlled problem
- `trajectories` -- (optional, default: `10`) number of simulated trajectories

### Output

An object of type `EnsembleSimulationSolution`.

### Notes

This function uses the ensemble simulations feature from
[`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl).
"""
function simulate(cp::AbstractControlProblem, args...; kwargs...)
    require(@__MODULE__, :OrdinaryDiffEq; fun_name="simulate")

    ivp = plant(cp)
    network = controller(cp)
    st_vars = states(cp)
    dist_vars = disturbances(cp)
    X₀ = project(initial_state(cp), st_vars)
    if !isempty(dist_vars)
        W₀ = project(initial_state(cp), dist_vars)
    end
    τ = period(cp)
    time_span = _get_tspan(args...; kwargs...)
    trajectories = get(kwargs, :trajectories, 10)
    inplace = get(kwargs, :inplace, true)
    preprocessing = control_preprocessing(cp)
    postprocessing = control_postprocessing(cp)
    include_vertices = get(kwargs, :include_vertices, false)

    t = tstart(time_span)
    iterations = ceil(Int, diam(time_span) / τ)

    # sample initial states
    x0_vec = sample(X₀, trajectories; include_vertices=include_vertices)
    trajectories = length(x0_vec)

    # preallocate
    extended = Vector{Vector{Float64}}(undef, trajectories)
    simulations = Vector{ODE.EnsembleSolution}(undef, iterations)
    all_controls = Vector{Vector{Vector{Float64}}}(undef, iterations)
    all_disturbances = Vector{Vector{Vector{Float64}}}(undef, iterations)

    @inbounds for i in 1:iterations
        # compute control inputs
        control_signals = Vector{Vector{Float64}}(undef, trajectories)
        for j in 1:trajectories
            x₀ = x0_vec[j]
            x₀ = apply(preprocessing, x₀)
            network_output = forward(x₀, network)
            control_signals[j] = apply(postprocessing, network_output)
        end
        all_controls[i] = control_signals

        # compute disturbances
        if !isempty(dist_vars)
            disturbance_signals = sample(W₀, trajectories)
            all_disturbances[i] = disturbance_signals
        else
            disturbance_signals = nothing
        end

        # extend system state with disturbances
        for j in 1:trajectories
            if isnothing(disturbance_signals)
                extended[j] = vcat(x0_vec[j], control_signals[j])
            else
                extended[j] = vcat(x0_vec[j], disturbance_signals[j], control_signals[j])
            end
        end

        T = i < iterations ? (t + τ) : tend(time_span)

        # simulate system for the next period
        simulations[i] = _solve_ensemble(ivp, extended, (t, T); inplace=inplace)

        # project to state variables
        for j in 1:trajectories
            ode_solution = simulations[i].u[j]
            final_extended = ode_solution.u[end]
            x0_vec[j] = final_extended[st_vars]
        end

        # advance time
        t = T
    end

    return EnsembleSimulationSolution(simulations, all_controls, all_disturbances)
end
