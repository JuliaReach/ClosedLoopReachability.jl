using ClosedLoopReachability, Test
import Aqua, ExplicitImports

@testset "ExplicitImports tests" begin
    ignores_all_explicit_imports_are_public = (:AbstractContinuousPost, :AbstractLazyReachSet,
                                               :AbstractTaylorModelReachSet, :ForwardAlgorithm,
                                               :ReachSolution, :TimeInterval, :_check_dim,
                                               :_default_cpost, :_get_tspan, :numtype, :post,
                                               :solve, :symBox, :zeroBox)
    ignores_all_qualified_accesses_are_public = [:inplace_field!, :outofplace_field]
    if v"1.10" <= VERSION < v"1.11"  # v1.10 was more strict with this
        push!(ignores_all_qualified_accesses_are_public, :get_extension)
    end
    ignores_all_qualified_accesses_are_public = Tuple(ignores_all_qualified_accesses_are_public)
    # due to reexporting ControllerFormats.FileFormats, NeuralNetworkReachability.ForwardAlgorithms,
    # and ReachabilityAnalysis
    ignores_no_implicit_imports = (:FileFormats, :ForwardAlgorithms, :ReachabilityAnalysis,
                                   :forward, :(..), :AbstractContinuousPost,
                                   :AbstractHyperrectangle, :Arrays, :BallInf,
                                   :BlackBoxContinuousSystem, :Flowpipe, :IVP, :Interval,
                                   :IntervalArithmetic, :IntervalBox, :LazySet, :LazySets,
                                   :MathematicalSystems, :MixedFlowpipe, :ReachabilityBase,
                                   :Taylor1, :TaylorModelReachSet, :TaylorN, :TaylorSeries,
                                   :ZeroSet, :Zonotope, :box_approximation, :cartesian_product,
                                   :center, :diam, :domain, :evaluate, :order, :high,
                                   :initial_state, :interval, :linear_map, :low, :mid, :ngens,
                                   :overapproximate, :polynomial, :project, :radius_hyperrectangle,
                                   :rand, :remainder, :rsetrep, :sample, :scale, :set, :variables!,
                                   :size, :sup, :system, :tend, :translate, :tstart, :(×), :shift,
                                   :vars)
    ignores_no_self_qualified_accesses = (:controller,)
    ExplicitImports.test_explicit_imports(ClosedLoopReachability;
                                          all_explicit_imports_are_public=(ignore=ignores_all_explicit_imports_are_public,),
                                          all_qualified_accesses_are_public=(ignore=ignores_all_qualified_accesses_are_public,),
                                          no_implicit_imports=(ignore=ignores_no_implicit_imports,),
                                          no_self_qualified_accesses=(ignore=ignores_no_self_qualified_accesses,))
end

@testset "Aqua tests" begin
    Aqua.test_all(ClosedLoopReachability)
end
