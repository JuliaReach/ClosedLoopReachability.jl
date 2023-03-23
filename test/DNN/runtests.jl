using ClosedLoopReachability.DNN
using Test

using ClosedLoopReachability.DNN: dim

import MAT, YAML

@testset "Architecture" begin
    @testset "DenseLayerOp" begin include("Architecture/DenseLayerOp.jl") end
    @testset "FeedforwardNetwork" begin include("Architecture/FeedforwardNetwork.jl") end
end

@testset "FileFormats" begin
    @testset "NNet" begin include("FileFormats/NNet.jl") end
    @testset "MAT" begin include("FileFormats/MAT.jl") end
    @testset "YAML" begin include("FileFormats/YAML.jl") end
    @testset "Sherlock" begin include("FileFormats/Sherlock.jl") end
end
