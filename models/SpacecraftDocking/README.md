# Spacecraft Docking

The controller was obtained from [here](https://github.com/verivital/ARCH-COMP2023/blob/540093e1d583c2c3a52b90ab293d684a659b7a49/benchmarks/Docking/model.mat)
and converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of the controller are equivalent.

```julia
import MAT
path = @modelpath("SpacecraftDocking", "model.mat")
controller_mat = read_MAT(path, act_key="act_fcns")
@assert controller_mat == controller
```
