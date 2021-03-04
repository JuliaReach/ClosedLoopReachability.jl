using .DifferentialEquations
const DE = DifferentialEquations

import .DifferentialEquations: controls

export trajectories,
       inputs

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
    @assert length(simulations) == length(controls) == length(inputs) "incompatible lengths"
    solutions = @inbounds [SimulationSolution(simulations[i], controls[i],
        isassigned(inputs, i) ? inputs[i] : nothing)
        for i in eachindex(simulations)]
    return EnsembleSimulationSolution(solutions)
end

Base.length(ess::EnsembleSimulationSolution) = length(ess.solutions)
solution(ess::EnsembleSimulationSolution, i) = ess.solutions[i]
trajectory(ess::EnsembleSimulationSolution, i) = trajectory(solution(ess, i))
trajectories(ess::EnsembleSimulationSolution) = trajectory.(ess.solutions)
controls(ess::EnsembleSimulationSolution, i) = controls(solution(ess, i))
inputs(ess::EnsembleSimulationSolution, i) = inputs(solution(ess, i))

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
