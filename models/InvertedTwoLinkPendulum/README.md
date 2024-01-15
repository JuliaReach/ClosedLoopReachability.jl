# Inverted Two-Link Pendulum

The controller for the less robust scenario was obtained from [here](https://github.com/verivital/ARCH-COMP2020/blob/bacba6e0c13a5220ae42baf2b88104d67e8856ac/benchmarks/Double_Pendulum/controller_double_pendulum_less_robust.nnet).

The controller for the more robust scenario was obtained from [here](https://github.com/verivital/ARCH-COMP2020/blob/bacba6e0c13a5220ae42baf2b88104d67e8856ac/benchmarks/Double_Pendulum/controller_double_pendulum_more_robust.nnet).

Both controllers were converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of each controller are equivalent.

```julia
path = @modelpath("InvertedTwoLinkPendulum", "controller_double_pendulum_less_robust.nnet")
controller_lr_nnet = read_NNet(path)
@assert controller_lr_nnet == controller_lr

path = @modelpath("InvertedTwoLinkPendulum", "controller_double_pendulum_more_robust.nnet")
controller_mr_nnet = read_NNet(path)
@assert controller_mr_nnet == controller_mr
```
