using .DifferentialEquations
const DE = DifferentialEquations

import .DifferentialEquations: controls

export trajectory,
       trajectories,
       inputs,
       solution

struct SimulationSolution{TT, CT, IT}
    trajectory::TT  # trajectory pieces for each control cycle
    controls::CT  # control inputs for each control cycle
    inputs::IT  # nondeterministic inputs for each control cycle
end

trajectory(sol::SimulationSolution) = sol.trajectory
controls(sol::SimulationSolution) = sol.controls
inputs(sol::SimulationSolution) = sol.inputs

struct EnsembleSimulationSolution{TT, CT, IT}
    solutions::Vector{SimulationSolution{TT, CT, IT}}
end

# constructor from a bulk input
function EnsembleSimulationSolution(simulations, controls, inputs)
    n = length(simulations)  # number of pieces
    m = length(simulations[1])  # number of trajectories
    @assert n == length(controls) == length(inputs) "incompatible lengths"
    @assert all(m == length(piece) for piece in simulations)

    simulations_new = @inbounds [[simulations[i][j] for i in 1:n] for j in 1:m]
    controls_new = @inbounds [[controls[i][j] for i in 1:n] for j in 1:m]
    inputs_new = @inbounds [[(isassigned(inputs, i) ? inputs[i][j] : nothing)
        for i in 1:n] for j in 1:m]
    solutions = @inbounds [SimulationSolution(simulations_new[j],
        controls_new[j], inputs_new[j]) for j in 1:m]
    return EnsembleSimulationSolution(solutions)
end

Base.length(ess::EnsembleSimulationSolution) = length(ess.solutions)
solution(ess::EnsembleSimulationSolution, i) = ess.solutions[i]
trajectory(ess::EnsembleSimulationSolution, i) = trajectory(solution(ess, i))
trajectories(ess::EnsembleSimulationSolution) = trajectory.(ess.solutions)
controls(ess::EnsembleSimulationSolution, i) = controls(solution(ess, i))
controls(ess::EnsembleSimulationSolution) = controls.(ess.solutions)
inputs(ess::EnsembleSimulationSolution, i) = inputs(solution(ess, i))
inputs(ess::EnsembleSimulationSolution) = inputs.(ess.solutions)

# simulation of multiple trajectories for an ODE system and a time span
# currently we can't use this method from RA because the sampling should be made from outside the function
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
