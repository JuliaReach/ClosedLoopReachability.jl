module OrdinaryDiffEqExt

import ClosedLoopReachability
import ClosedLoopReachability.ReachabilityAnalysis as RA

@static if isdefined(Base, :get_extension)
    import OrdinaryDiffEq
else
    import .OrdinaryDiffEq
end
const ODE = OrdinaryDiffEq

if isdefined(OrdinaryDiffEq, :controls)
    # before v7, DE had deps importing ModelingToolkit, which exports `controls`
    @static if isdefined(Base, :get_extension)
        import OrdinaryDiffEq: controls
    else
        import .OrdinaryDiffEq: controls
    end
end

function ClosedLoopReachability._initialize_simulation_container(iterations::Int)
    return Vector{ODE.EnsembleSolution}(undef, iterations)
end

# simulation of multiple trajectories for an ODE system and a time span
# currently we can't use this method from RA because the sampling should be made from outside the function
function ClosedLoopReachability._solve_ensemble(ivp, X0_samples::Vector, tspan;
                                                trajectories_alg=ODE.Tsit5(),
                                                ensemble_alg=ODE.EnsembleThreads(),
                                                inplace=true,
                                                kwargs...)
    if inplace
        field = RA.inplace_field!(ivp)  # NOTE: this is an internal function
    else
        field = RA.outofplace_field(ivp)  # NOTE: this is an internal function
    end

    # the third argument `repeat` is not needed here
    _prob_func(prob, i, _) = ODE.remake(prob; u0=X0_samples[i])
    ensemble_prob = ODE.EnsembleProblem(ODE.ODEProblem(field, first(X0_samples), tspan);
                                        prob_func=_prob_func)
    return ODE.solve(ensemble_prob, trajectories_alg, ensemble_alg;
                     trajectories=length(X0_samples))
end

end  # module
