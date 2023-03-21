using ClosedLoopReachability.DNN
using Test

using ClosedLoopReachability.DNN: dim

import MAT

@testset "Architecture" begin
    @testset "DenseLayerOp" begin include("Architecture/DenseLayerOp.jl") end
    @testset "FeedforwardNetwork" begin include("Architecture/FeedforwardNetwork.jl") end
end

@testset "FileFormats" begin
    @testset "NNet" begin include("FileFormats/NNet.jl") end
    @testset "MAT" begin include("FileFormats/MAT.jl") end
end
