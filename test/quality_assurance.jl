using ClosedLoopReachability, Test
import Aqua, ExplicitImports

@testset "ExplicitImports tests" begin
    ignores = (:AbstractContinuousPost, :AbstractLazyReachSet,
               :AbstractTaylorModelReachSet, :ForwardAlgorithm, :ReachSolution,
               :_check_dim, :_default_cpost, :_get_tspan, :numtype, :post,
               :solve, :symBox, :zeroBox)
    @test isnothing(ExplicitImports.check_all_explicit_imports_are_public(ClosedLoopReachability;
                                                                          ignore=ignores))
    @test isnothing(ExplicitImports.check_all_explicit_imports_via_owners(ClosedLoopReachability))
    ignores = (:inplace_field!, :outofplace_field)
    @test isnothing(ExplicitImports.check_all_qualified_accesses_are_public(ClosedLoopReachability;
                                                                            ignore=ignores))
    @test isnothing(ExplicitImports.check_all_qualified_accesses_via_owners(ClosedLoopReachability))
    # due to reexporting ControllerFormats.FileFormats, NeuralNetworkReachability.ForwardAlgorithms,
    # and ReachabilityAnalysis
    ignores = (:FileFormats, :ForwardAlgorithms, :ReachabilityAnalysis,
               :forward, :(..), :AbstractContinuousPost,
               :AbstractHyperrectangle, :Arrays, :BallInf,
               :BlackBoxContinuousSystem, :Flowpipe, :IVP, :Interval,
               :IntervalArithmetic, :IntervalBox, :LazySet, :LazySets,
               :MathematicalSystems, :MixedFlowpipe, :ReachabilityBase,
               :Taylor1, :TaylorModelReachSet, :TaylorN, :ZeroSet, :Zonotope,
               :box_approximation, :cartesian_product, :diam, :domain,
               :evaluate, :get_order, :high, :initial_state, :interval,
               :linear_map, :low, :mid, :ngens, :overapproximate, :polynomial,
               :project, :rand, :remainder, :rsetrep, :sample, :scale, :set,
               :set_variables, :size, :sup, :system, :tend, :translate, :tstart,
               :(×), :shift, :vars)
    @test isnothing(ExplicitImports.check_no_implicit_imports(ClosedLoopReachability;
                                                              ignore=ignores))
    ignores = (:controller,)
    @test isnothing(ExplicitImports.check_no_self_qualified_accesses(ClosedLoopReachability;
                                                                     ignore=ignores))
    @test isnothing(ExplicitImports.check_no_stale_explicit_imports(ClosedLoopReachability))
end

@testset "Aqua tests" begin
    Aqua.test_all(ClosedLoopReachability)
end
