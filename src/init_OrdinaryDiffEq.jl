import .OrdinaryDiffEq
const ODE = OrdinaryDiffEq

if isdefined(OrdinaryDiffEq, :controls)
    # before v7, DE had deps importing ModelingToolkit, which exports `controls`
    import .OrdinaryDiffEq: controls
end

export trajectory,
       trajectories,
       controls,
       disturbances,
       solution

struct SimulationSolution{TT,CT,IT}
    trajectory::TT  # trajectory pieces for each control cycle
    controls::CT  # control inputs for each control cycle
    disturbances::IT  # disturbances for each control cycle
end

trajectory(sol::SimulationSolution) = sol.trajectory
controls(sol::SimulationSolution) = sol.controls
disturbances(sol::SimulationSolution) = sol.disturbances

struct EnsembleSimulationSolution{TT,CT,IT}
    solutions::Vector{SimulationSolution{TT,CT,IT}}
end

# constructor from a bulk input
function EnsembleSimulationSolution(simulations, controls, disturbances)
    n = length(simulations)  # number of pieces
    m = length(simulations[1])  # number of trajectories
    @assert n == length(controls) == length(disturbances) "incompatible lengths"
    @assert all(m == length(piece) for piece in simulations)

    simulations_new = @inbounds [[simulations[i][j] for i in 1:n] for j in 1:m]
    controls_new = @inbounds [[controls[i][j] for i in 1:n] for j in 1:m]
    disturbances_new = @inbounds [[(isassigned(disturbances, i) ? disturbances[i][j] : nothing)
                                   for i in 1:n] for j in 1:m]
    solutions = @inbounds [SimulationSolution(simulations_new[j],
                                              controls_new[j], disturbances_new[j]) for j in 1:m]
    return EnsembleSimulationSolution(solutions)
end

Base.length(ess::EnsembleSimulationSolution) = length(ess.solutions)
solution(ess::EnsembleSimulationSolution, i) = ess.solutions[i]
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
