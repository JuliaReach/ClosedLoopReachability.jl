using DifferentialEquations
const DE = DifferentialEquations

function simulate(cp::AbstractNeuralNetworkControlProblem, args...; kwargs...)
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

# simulation of multiple trajectories for an ODE system and a time span
function _solve_ensemble(ivp, X0_samples, tspan;
                         trajectories_alg=DE.Tsit5(),
                         ensemble_alg=DE.EnsembleThreads(),
                         inplace=true,
                         kwargs...)
    if inplace
        field = ReachabilityAnalysis.inplace_field!(ivp)
    else
        field = ReachabilityAnalysis.outofplace_field(ivp)
    end

    _prob_func(prob, i, repeat) = remake(prob, u0=X0_samples[i])
    ensemble_prob = EnsembleProblem(ODEProblem(field, first(X0_samples), tspan),
                                    prob_func=_prob_func)
    return DE.solve(ensemble_prob, trajectories_alg, ensemble_alg;
                    trajectories=length(X0_samples))
end
