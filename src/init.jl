using Requires: @require
using Reexport: @reexport

using Base: isempty
using ControllerFormats: ActivationFunction, DenseLayerOp, FeedforwardNetwork,
                         Id, ReLU
# controller formats and parsers
@reexport using ControllerFormats.FileFormats

@reexport using NeuralNetworkReachability.ForwardAlgorithms
using NeuralNetworkReachability.ForwardAlgorithms: ForwardAlgorithm

@reexport using ReachabilityAnalysis
# namespace conflict
using ReachabilityAnalysis: dim
# unexported methods
using ReachabilityAnalysis: _check_dim, _get_tspan, _default_cpost,  # NOTE: these are internal symbols
                            ReachSolution, InitialValueProblem, numtype, post,
                            AbstractContinuousPost, AbstractLazyReachSet,
                            AbstractTaylorModelReachSet, zeroBox, symBox
using ReachabilityAnalysis.TM: TaylorModel1, TaylorModelN, fp_rpa

using ReachabilityBase.Require: require
using ReachabilityBase.Comparison: isapproxzero

using Parameters: @with_kw

# aliases
const RA = ReachabilityAnalysis
const IA = IntervalArithmetic

import CommonSolve: solve

# optional dependencies
function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        include("init_OrdinaryDiffEq.jl")
        @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
            import .Plots
            include("init_OrdinaryDiffEq_Plots.jl")
        end
    end
end
