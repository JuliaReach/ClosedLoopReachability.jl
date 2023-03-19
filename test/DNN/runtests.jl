using ClosedLoopReachability.DNN
using Test

using ClosedLoopReachability.DNN: dim

@testset "Architecture" begin
    @testset "DenseLayerOp" begin include("Architecture/DenseLayerOp.jl") end
    @testset "FeedforwardNetwork" begin include("Architecture/FeedforwardNetwork.jl") end
end
