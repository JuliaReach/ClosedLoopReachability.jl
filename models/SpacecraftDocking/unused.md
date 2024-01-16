# Original radius:
r = [18, 18, 0.28, 0.28]

# Property check for simulation:
using ClosedLoopReachability: EnsembleSimulationSolution
function predicate_set(sim_sols::EnsembleSimulationSolution)
    for sim in sim_sols.solutions
        for traj in sim.trajectory
            for i in eachindex(traj)
                if !predicate_point(traj[i])
                    return false
                end
            end
        end
    end
    return true
end
