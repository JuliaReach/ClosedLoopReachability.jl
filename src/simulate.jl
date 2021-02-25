"""
    simulate(cp::AbstractNeuralNetworkControlProblem, args...; kwargs...)

Simulate a neural-network controlled system for a family of random trajectories.

### Input

- `cp`           -- neural network controlled problem
- `trajectories` -- (optional, default: `10`) number of simulated trajectories

### Output

The tuple `(simulations, all_controls)` which contains the vector with each simulation
and the vector of controls used.

### Notes

This function uses the ensemble simulations feature from `DifferentialEquations.jl`.
"""
function simulate(cp::AbstractNeuralNetworkControlProblem, args...; kwargs...)
    if !isdefined(@__MODULE__, :DifferentialEquations)
        error("package 'DifferentialEquations' is required for simulation")
    end

    ivp = plant(cp)
    network = controller(cp)
    st_vars = state_vars(cp)
    n = length(st_vars)
    X₀ = Projection(initial_state(ivp), st_vars)
    ctrl_vars = control_vars(cp)
    τ = period(cp)
    time_span = _get_tspan(args...; kwargs...)
    trajectories = get(kwargs, :trajectories, 10)
    inplace = get(kwargs, :inplace, true)

    t = tstart(time_span)
    iterations = ceil(Int, diam(time_span) / τ)

    # preallocate
    extended = Vector{Vector{Float64}}(undef, trajectories)
    simulations = Vector{EnsembleSolution}(undef, iterations)
    all_controls = Vector{Vector{Vector{Float64}}}(undef, iterations)

    # sample initial states
    states = sample(X₀, trajectories)

    @inbounds for i in 1:iterations
        # compute control inputs
        controls = Vector{Vector{Float64}}(undef, trajectories)
        for j in 1:trajectories
            x₀ = states[j]
            controls[j] = forward(network, x₀)
        end
        all_controls[i] = controls

        # extend system state with inputs
        for j in 1:trajectories
            extended[j] = vcat(states[j], controls[j])
        end

        # simulate system for the next period
        simulations[i] = _solve_ensemble(ivp, extended, (t, t + τ);
                                         inplace=inplace)

        # project to state variables
        for j in 1:trajectories
            ode_solution = simulations[i][j]
            final_extended = ode_solution.u[end]
            states[j] = final_extended[st_vars]
        end

        # advance time
        t += τ
    end

    return simulations, all_controls
end
