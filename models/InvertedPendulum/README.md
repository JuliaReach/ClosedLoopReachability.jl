# Inverted Pendulum

The controller was obtained from [here](https://github.com/verivital/ARCH-COMP2020/blob/cd192b0a21088fbb26fe018c847424b659129752/benchmarks/Single_Pendulum/controller_single_pendulum.nnet)
and converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of the controller are equivalent.

```julia
path = @modelpath("InvertedPendulum", "controller_single_pendulum.nnet")
controller_nnet = read_NNet(path)
@assert controller_nnet == controller
```
