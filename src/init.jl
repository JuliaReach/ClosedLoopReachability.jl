@static if !isdefined(Base, :get_extension)
    using Requires: @require
end
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
using ReachabilityAnalysis: AbstractContinuousPost, AbstractLazyReachSet,  # NOTE: these are internal symbols
                            AbstractTaylorModelReachSet, InitialValueProblem,
                            TimeInterval, ReachSolution, numtype, post, symBox,
                            zeroBox, _check_dim, _default_cpost, _get_tspan
using ReachabilityBase.Require: require
using ReachabilityBase.Comparison: isapproxzero
using TaylorModels: TaylorModel1, TaylorModelN, fp_rpa

using Parameters: @with_kw

# aliases
const RA = ReachabilityAnalysis
const IA = IntervalArithmetic

import CommonSolve: solve

# optional dependencies
@static if !isdefined(Base, :get_extension)
    function __init__()
        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            include("../ext/OrdinaryDiffEqExt.jl")
            @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
                include("../ext/OrdinaryDiffEqPlotsExt.jl")
            end
        end
    end
end
