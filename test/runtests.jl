using ClosedLoopReachability
using Test

@testset "DNN" begin include("DNN/runtests.jl") end

@testset "Toy model (black-box network)" begin include("black_box_toy_model.jl") end
