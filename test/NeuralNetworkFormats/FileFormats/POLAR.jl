# valid sample file
file = joinpath(@__DIR__, "sample_POLAR")

# parse file
N = read_POLAR(file)

@test length(N.layers) == 4
