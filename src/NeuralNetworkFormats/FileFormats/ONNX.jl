"""
    read_ONNX(filename::String; [input_dimension=nothing])

Read a neural network stored in [ONNX](https://github.com/onnx/onnx) format.
This function requires to load the
[`ONNX.jl` library](https://github.com/FluxML/ONNX.jl).

### Input

- `filename`        -- name of the `ONNX` file
- `input_dimension` -- (optional; default: `nothing`) input dimension (required
                       by `ONNX.jl` parser); see the notes below

### Output

A [`FeedforwardNetwork`](@ref).

### Notes

This implementation assumes the following structure:
1. First comes the input vector (which is ignored).
2. Next come the weight matrices `W` (transposed) and bias vectors `b` in pairs
   *in the order in which they are applied*.
3. Next come the affine maps and the activation functions *in the order in which
   they are applied*. The last layer does not have an activation function.

Some of these assumptions are currently *not validated*. Hence it may happen
that this function returns a result that is incorrect.

If the argument `input_dimension` is not provided, the file is parsed an
additional time to read the correct number (which is inefficient).
"""
function read_ONNX(filename::String; input_dimension=nothing)
    require(@__MODULE__, :ONNX; fun_name="read_ONNX")

    # parse input dimension if not provided
    if isnothing(input_dimension)
        open(filename) do io
            onnx_raw_model = ONNX.decode(ONNX.ProtoDecoder(io), ONNX.ModelProto)
            input = onnx_raw_model.graph.input
            @assert input isa Vector{ONNX.ValueInfoProto} && length(input) == 1
            dimensions = input[1].var"#type".value.value.shape.dim
            @assert dimensions isa Vector{ONNX.var"TensorShapeProto.Dimension"} &&
                    length(dimensions) == 2 && dimensions[1].value.value == 1
            input_dimension = dimensions[2].value.value
        end
    end

    # ONNX.jl expects an input, so the user must provide that
    x0 = zeros(Float32, input_dimension)

    # read data
    data = load(filename, x0)

    @assert data isa Umlaut.Tape{ONNX.ONNXCtx} "`read_ONNX` must be called " *
        "with `ONNX.Umlaut.Tape{ONNX.ONNXCtx}`"

    layer_parameters = []
    ops = data.ops
    @assert ops[1] isa Umlaut.Input && iszero(ops[1].val)  # skip input operation
    idx = 2
    @inbounds while idx <= length(ops)
        op = ops[idx]
        if !(op.val isa AbstractMatrix)
            break
        end
        W = permutedims(op.val)
        idx += 1
        op = ops[idx]
        @assert op.val isa AbstractVector "expected a bias vector"
        b = op.val
        push!(layer_parameters, (W, b))
        idx += 1
    end
    n_layers = div(idx - 2, 2)
    @assert length(ops) == 4 * n_layers
    T = DenseLayerOp{<:ActivationFunction, Matrix{Float32}, Vector{Float32}}
    layers = T[]
    layer = 1
    while idx <= length(ops)
        # affine map (treated implicitly)
        op = ops[idx]
        @assert op isa Umlaut.Call "expected an affine map"
        args = op.args
        @assert length(args) == 5
        @assert args[2] == ONNX.onnx_gemm
        @assert args[3]._op.id == (layer == 1 ? 1 : idx - 1)
        @assert args[4]._op.id == 2 * layer
        @assert args[5]._op.id == 2 * layer + 1
        W, b = @inbounds layer_parameters[layer]
        idx += 1

        # activation function
        if idx > length(ops)
            # last layer is assumed to be the identity
            a = Id()
        else
            op = ops[idx]
            @assert op isa Umlaut.Call "expected an activation function"
            args = op.args
            @assert length(args) == 2
            @assert args[2]._op.id == idx - 1
            a = available_activations[string(args[1])]
            idx += 1
        end

        L = DenseLayerOp(W, b, a)
        push!(layers, L)
        layer += 1
    end

    return FeedforwardNetwork(layers)
end
