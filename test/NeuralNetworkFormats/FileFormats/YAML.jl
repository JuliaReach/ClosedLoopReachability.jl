# valid sample file
file = joinpath(@__DIR__, "sample_YAML.yml")

# parse file
N = read_YAML(file)

@test length(N.layers) == 4
