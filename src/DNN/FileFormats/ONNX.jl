# ========================================
# ONNX format
# (load the data with the ONNX.jl package)
# ========================================

const ACT_ONNX = Dict("σ"=>Sigmoid())

"""
    read_nnet_onnx(data::Dict)

Read a neural network from a file in ONNX format (see `ONNX.jl`) and convert it.

### Input

- `data` -- an `ONNXCtx` struct parsed by `ONNX.jl`

### Output

A `FeedforwardNetwork` struct.

### Notes

This implementation does not apply to general `ONNX` networks because it assumes
a specific structure:
1. First comes a bias vector for the input vector that is a zero vector.
2. Next come the weight matrices `W` (transposed) and bias vectors `b` in pairs
*in the order in which they are applied*.
3. Next come the affine maps and the activation functions *in the order in which
they are applied*. The last layer does not have an activation function.

Some of these assumptions *are not checked*. Hence it may happen that it returns
a result that is incorrect. A general implementation may be added in the future.

The following activation function is supported: sigmoid (and an implicit
identity in the last layer); see `ClosedLoopReachability.ACT_ONNX`.
"""
function read_nnet_onnx(data)
    require(@__MODULE__, :ONNX; fun_name="read_nnet_onnx")

    @assert data isa Ghost.Tape{ONNX.ONNXCtx} "`read_nnet_onnx` must be " *
        "called with `ONNX.Ghost.Tape{ONNX.ONNXCtx}`"

    layer_parameters = []
    ops = data.ops
    @assert ops[1] isa Ghost.Input && iszero(ops[1].val)  # skip input operation
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
        @assert op isa Ghost.Call "expected an affine map"
        args = op.args
        @assert length(args) == 5
        @assert args[2] == onnx_gemm
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
            @assert op isa Ghost.Call "expected an activation function"
            args = op.args
            @assert length(args) == 2
            @assert args[2]._op.id == idx - 1
            a = ACT_ONNX[string(args[1])]
            idx += 1
        end

        L = DenseLayerOp(W, b, a)
        push!(layers, L)
        layer += 1
    end
    return FeedforwardNetwork(layers)
end
