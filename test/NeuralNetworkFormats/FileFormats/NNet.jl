# valid sample file; features:
# - comma-separated lists may contain whitespaces
# - lines may end with a whitespace
# - line 7 contains one element too much, which must be ignored
file = joinpath(@__DIR__, "sample_NNet.nnet")

# parse file
N = read_NNet(file)

@test length(N.layers) == 3
