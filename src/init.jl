using Requires, Reexport

using ControllerFormats
# namespace conflict
using ControllerFormats: Id
# parsers
@reexport using ControllerFormats.FileFormats

@reexport using NeuralNetworkReachability.ForwardAlgorithms
using NeuralNetworkReachability.ForwardAlgorithms: ForwardAlgorithm

@reexport using ReachabilityAnalysis
# namespace conflict
using ReachabilityAnalysis: dim
# unexported methods
using ReachabilityAnalysis: _check_dim, _get_tspan, _get_cpost, _default_cpost,
                            ReachSolution, InitialValueProblem, numtype, post,
                            AbstractContinuousPost, TimeInterval,
                            AbstractLazyReachSet, AbstractTaylorModelReachSet

using ReachabilityBase.Require
using ReachabilityBase.Arrays: SingleEntryVector
using ReachabilityBase.Comparison: isapproxzero

using Parameters: @with_kw

# aliases
const RA = ReachabilityAnalysis
const IA = IntervalArithmetic

import CommonSolve: solve

# optional dependencies
function __init__()
    @require DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa" begin
        include("init_DifferentialEquations.jl")
        @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
            import .Plots
            include("init_DifferentialEquations_Plots.jl")
        end
    end
end
