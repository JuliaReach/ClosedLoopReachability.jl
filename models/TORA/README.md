# Translational Oscillations by a Rotational Actuator (TORA)

The ReLU controller was obtained from [here](https://github.com/verivital/ARCH-COMP2024/blob/92a32d711a136bebc8f7d796140ef80323ce8272/benchmarks/Benchmark9-Tora/controllerTora.mat).

The ReLU/tanh controller was obtained from [here](https://github.com/verivital/ARCH-COMP2022/blob/9edf874ffe2a4631a333ac3f61509243f33faadf/benchmarks/Tora_Heterogeneous/nn_tora_relu_tanh.mat).

The sigmoid controller was obtained from [here](https://github.com/verivital/ARCH-COMP2022/blob/9edf874ffe2a4631a333ac3f61509243f33faadf/benchmarks/Tora_Heterogeneous/nn_tora_sigmoid.mat).

All controllers were converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of each controller are equivalent.

```julia
import MAT

path = @modelpath("TORA", "controllerTora.mat")
controller_mat = read_MAT(path, act_key="act_fcns")
@assert controller_mat == controller

path = @modelpath("TORA", "nn_tora_relu_tanh.mat")
controller_relutanh_mat = read_MAT(path, act_key="act_fcns")
@assert controller_relutanh_mat == controller_relutanh

path = @modelpath("TORA", "nn_tora_sigmoid.mat")
controller_sigmoid_mat = read_MAT(path, act_key="act_fcns")
@assert controller_sigmoid_mat == controller_sigmoid
```
