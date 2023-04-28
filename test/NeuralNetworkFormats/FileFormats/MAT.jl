# valid sample file
file = joinpath(@__DIR__, "sample_MAT.mat")

# parse file
N = read_MAT(file, act_key="act_fcns")

@test length(N.layers) == 4
