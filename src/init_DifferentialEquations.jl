using .DifferentialEquations
const DE = DifferentialEquations

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
