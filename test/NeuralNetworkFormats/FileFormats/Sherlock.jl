# valid sample file from Sherlock manual
file = joinpath(@__DIR__, "sample_Sherlock")

# parse file
N = read_Sherlock(file)

@test dim_in(N) == 2
@test dim_out(N.layers[1]) == 2
@test dim_out(N) == 1
