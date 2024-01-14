# Airplane

The controller was obtained from [here](https://github.com/verivital/ARCH-COMP2020/blob/f7ba48de5b20cc818a9c1154d1efa98dda0b88ff/benchmarks/Airplane/controller_airplane.nnet)
and converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of the controller are equivalent.

```julia
path = @modelpath("Airplane", "controller_airplane.nnet")
controller_nnet = read_NNet(path)
@assert controller_nnet == controller
```
