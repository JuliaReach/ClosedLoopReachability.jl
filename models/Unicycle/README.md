# Unicycle

The controller was obtained from [here](https://github.com/verivital/ARCH-COMP2020/blob/f8d91e7009c7a4ee698747f6e88cd4d2987b16fe/benchmarks/Benchmark10-Unicycle/controllerB_nnv.mat)
and converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of the controller are numerically
close.

```julia
import MAT
path = @modelpath("Unicycle", "controllerB_nnv.mat")
controller_mat = read_MAT(path; act_key="act_fcns")
@assert ≈(controller_mat, controller; rtol=1e-7)
```
