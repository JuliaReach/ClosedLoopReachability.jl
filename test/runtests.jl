using ClosedLoopReachability
using Test

@testset "NeuralNetworkFormats" begin include("NeuralNetworkFormats/runtests.jl") end

@testset "Toy model (black-box network)" begin include("black_box_toy_model.jl") end
