module OrdinaryDiffEqExt

@static if isdefined(Base, :get_extension)
    import OrdinaryDiffEq
    using ClosedLoopReachability: SimulationSolution, EnsembleSimulationSolution
else
    import .OrdinaryDiffEq
    using ..SimulationSolution
    using ..EnsembleSimulationSolution
end
const ODE = OrdinaryDiffEq


if isdefined(OrdinaryDiffEq, :controls)
    # before v7, DE had deps importing ModelingToolkit, which exports `controls`
    import .OrdinaryDiffEq: controls
end

export trajectory,
       trajectories,
       controls,
       disturbances,
       solutions

Base.length(sol::SimulationSolution) = length(sol.trajectory)
function Base.getindex(sol::SimulationSolution, i)
    return SimulationSolution(sol.trajectory[i],
                              sol.controls[i],
                              sol.disturbances[i])
end
trajectory(sol::SimulationSolution) = sol.trajectory
controls(sol::SimulationSolution) = sol.controls
disturbances(sol::SimulationSolution) = sol.disturbances


# constructor from a bulk input
function EnsembleSimulationSolution(simulations, controls, disturbances)
    n = length(simulations)  # number of pieces
    m = length(simulations[1])  # number of trajectories
    @assert n == length(controls) == length(disturbances) "incompatible lengths"
    @assert all(m == length(piece) for piece in simulations)

    simulations_new = @inbounds [[simulations[i].u[j] for i in 1:n] for j in 1:m]
    controls_new = @inbounds [[controls[i][j] for i in 1:n] for j in 1:m]
    disturbances_new = @inbounds [[(isassigned(disturbances, i) ? disturbances[i][j] : nothing)
                                   for i in 1:n] for j in 1:m]
    solutions = @inbounds [SimulationSolution(simulations_new[j],
                                              controls_new[j], disturbances_new[j]) for j in 1:m]
    return EnsembleSimulationSolution(solutions)
end

Base.length(ess::EnsembleSimulationSolution) = length(ess.solutions)
Base.getindex(ess::EnsembleSimulationSolution, i) = ess.solutions[i]
function solutions(ess::EnsembleSimulationSolution, i)
    return EnsembleSimulationSolution([sol[i] for sol in ess.solutions])
end
trajectory(ess::EnsembleSimulationSolution, i) = trajectory(solution(ess, i))
trajectories(ess::EnsembleSimulationSolution) = trajectory.(ess.solutions)
controls(ess::EnsembleSimulationSolution, i) = controls(solution(ess, i))
controls(ess::EnsembleSimulationSolution) = controls.(ess.solutions)
disturbances(ess::EnsembleSimulationSolution, i) = disturbances(solution(ess, i))
disturbances(ess::EnsembleSimulationSolution) = disturbances.(ess.solutions)

# simulation of multiple trajectories for an ODE system and a time span
# currently we can't use this method from RA because the sampling should be made from outside the function
function _solve_ensemble(ivp, X0_samples, tspan;
                         trajectories_alg=ODE.Tsit5(),
                         ensemble_alg=ODE.EnsembleThreads(),
                         inplace=true,
                         kwargs...)
    if inplace
        field = ReachabilityAnalysis.inplace_field!(ivp)
    else
        field = ReachabilityAnalysis.outofplace_field(ivp)
    end

    # the third argument `repeat` is not needed here
    _prob_func(prob, i, _) = ODE.remake(prob; u0=X0_samples[i])
    ensemble_prob = ODE.EnsembleProblem(ODE.ODEProblem(field, first(X0_samples), tspan);
                                        prob_func=_prob_func)
    return ODE.solve(ensemble_prob, trajectories_alg, ensemble_alg;
                     trajectories=length(X0_samples))
end

function _simulate(cp::AbstractControlProblem, args...; kwargs...)
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
        simulations[i] = _solve_ensemble(ivp, extended, (t, T);
                                         inplace=inplace)

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

end  # module
