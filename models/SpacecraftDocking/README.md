# Spacecraft Docking

The controller was obtained from [here](https://github.com/verivital/ARCH-COMP2022/blob/df48a650a9119c1013a6c51dc32d9d6a21480e0c/benchmarks/Docking/model.mat)
and converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of the controller are equivalent.

```julia
import MAT
path = @modelpath("SpacecraftDocking", "model.mat")
controller_mat = read_MAT(path, act_key="act_fcns")
@assert controller_mat == controller
```
