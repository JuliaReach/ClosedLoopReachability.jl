# Attitude Control

The controller was obtained from [here](https://github.com/verivital/ARCH-COMP2022/blob/3091118ce3a1e4177c4e5796e956885ac6f7695f/benchmarks/Aircraft/Attitude%20Control/attitude_control_3_64_torch.onnx)
and converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of the controller are equivalent.

```julia
import ONNX
path = @modelpath("AttitudeControl", "attitude_control_3_64_torch.onnx")
controller_onnx = read_ONNX(path; input_dimension=6)
@assert controller_onnx == controller
```
