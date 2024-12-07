struct SimulationSolution{TT,CT,IT}
    trajectory::TT  # trajectory pieces for each control cycle
    controls::CT  # control inputs for each control cycle
    disturbances::IT  # disturbances for each control cycle
end

Base.length(sol::SimulationSolution) = length(sol.trajectory)
function Base.getindex(sol::SimulationSolution, i)
    return SimulationSolution(sol.trajectory[i],
                              sol.controls[i],
                              sol.disturbances[i])
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
    mod = isdefined(Base, :get_extension) ?
          Base.get_extension(@__MODULE__, :OrdinaryDiffEqExt) : @__MODULE__
    require(mod, :OrdinaryDiffEq; fun_name="simulate")
    return mod._simulate(cp, args...; kwargs...)
end
