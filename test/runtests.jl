using ClosedLoopReachability
using Test

@time @testset "Toy model (black-box network)" begin include("black_box_toy_model.jl") end
