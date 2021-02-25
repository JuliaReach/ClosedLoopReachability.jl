using Reexport

function __init__()
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" include("init_DifferentialEquations.jl")
end

@reexport using ReachabilityAnalysis
using ReachabilityAnalysis: _check_dim, _get_tspan, _get_cpost, _default_cpost,
                            ReachSolution, InitialValueProblem, numtype,
                            AbstractContinuousPost, TimeInterval,
                            AbstractLazyReachSet, AbstractTaylorModelReachSet,
                            complement # conflict with NeuralVerification.jl

const RA = ReachabilityAnalysis
import ReachabilityAnalysis: solve

@reexport using NeuralVerification
using NeuralVerification: Network, output_bound, Solver, Id, ReLU

const NV = NeuralVerification
const IA = IntervalArithmetic

using LazySets: _leq, _geq, isapproxzero, remove_zero_generators
