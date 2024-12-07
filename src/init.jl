# optional dependencies
using PackageExtensionCompat
function __init__()
    @require_extensions
end
function plot_simulation! end
export plot_simulation!

using Reexport

using ControllerFormats
# namespace conflict
using ControllerFormats: Id
# controller formats and parsers
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
                            AbstractLazyReachSet, AbstractTaylorModelReachSet,
                            TaylorModel1, TaylorModelN, fp_rpa, zeroBox, symBox

using ReachabilityBase.Require
using ReachabilityBase.Arrays: SingleEntryVector
using ReachabilityBase.Comparison: isapproxzero

using Parameters: @with_kw

# aliases
const RA = ReachabilityAnalysis
const IA = IntervalArithmetic

import CommonSolve: solve
