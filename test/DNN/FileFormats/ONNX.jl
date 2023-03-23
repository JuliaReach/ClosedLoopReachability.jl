# valid sample file
file = joinpath(@__DIR__, "sample_ONNX.onnx")

# parse file
N = read_ONNX(file);

# alternative parse with optional argument
N2 = read_ONNX(file; input_dimension=6);
@test N == N2

@test length(N.layers) == 4
