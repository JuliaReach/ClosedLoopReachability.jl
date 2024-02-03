# Adaptive Cruise Controller (ACC)

The ReLU controller was obtained from [here](https://github.com/verivital/ARCH-COMP2020/blob/31d517b3ff41f2271170f978447211e5ebfe14ed/benchmarks/ACC/controller_5_20.mat).

The tanh controller was obtained from [here](https://gitlab.com/goranf/ARCH-COMP/-/blob/59d73ef5d40fbca3f2f8bbf785ad0ae9261376b8/2019/AINNCS/verisig/benchmarks/ACC/tanh.yml).

Both controllers were converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of each controller are equivalent
respectively numerically close (the YAML format uses `Float32` representation).

```julia
import MAT
path = @modelpath("ACC", "controller_5_20.mat")
controller_relu_mat = read_MAT(path; act_key="act_fcns")
@assert controller_relu_mat == controller_relu

import YAML
path = @modelpath("ACC", "tanh.yml")
controller_tanh_yml = read_YAML(path)
@assert ≈(controller_tanh_yml, controller_tanh; rtol=Base.rtoldefault(Float32))
```
