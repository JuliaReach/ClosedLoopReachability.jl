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
    mod = isdefined(Base, :get_extension) ?
          Base.get_extension(@__MODULE__, :OrdinaryDiffEqExt) : @__MODULE__
    require(mod, :OrdinaryDiffEq; fun_name="simulate")
    return mod._simulate(cp, args...; kwargs...)
end
