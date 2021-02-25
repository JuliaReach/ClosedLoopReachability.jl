using Requires, Reexport

@reexport using ReachabilityAnalysis

# unexported methods
using ReachabilityAnalysis: _check_dim, _get_tspan, _get_cpost, _default_cpost,
                            ReachSolution, InitialValueProblem, numtype,
                            AbstractContinuousPost, TimeInterval,
                            AbstractLazyReachSet, AbstractTaylorModelReachSet

@reexport using NeuralVerification
using NeuralVerification: Network, output_bound, Solver, Id, ReLU

# aliases
const RA = ReachabilityAnalysis
const NV = NeuralVerification
const IA = IntervalArithmetic

using LazySets: _leq, _geq, isapproxzero, remove_zero_generators

# resolve conflicts
using ReachabilityAnalysis: solve, complement

# optional dependencies
function __init__()
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" include("init_DifferentialEquations.jl")
end
