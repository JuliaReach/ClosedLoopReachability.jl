# optional dependencies
function __init__()
    @require MAT = "23992714-dd62-5051-b70f-ba57cb901cac" begin
        using .MAT: matread
    end

    @require ONNX = "d0dd6a25-fac6-55c0-abf7-829e0c774d20" begin
        using .ONNX: ONNXCtx, Ghost, onnx_gemm
    end
end
