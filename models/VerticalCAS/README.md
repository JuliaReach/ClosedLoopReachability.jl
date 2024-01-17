# Vertical Collision Avoidance System (VerticalCAS)

The controllers were obtained from [here](https://github.com/verivital/ARCH-COMP2020/tree/3473769f390720440964fa1e460883585d575030/benchmarks/VCAS/nnet_networks).

All controllers were converted to POLAR format using [ControllerFormats.jl](https://github.com/JuliaReach/ControllerFormats.jl).

The script below asserts that the two versions of each controller are equivalent.

```julia
path_prefix = @modelpath("VerticalCAS", "")
for i = 1:9
    path = joinpath(path_prefix, "VerticalCAS_controller_$(i).polar")
    controller = read_POLAR(path)
    path = joinpath(path_prefix, "VertCAS_noResp_pra0$(i)_v9_20HU_200.nnet")
    controller_nnet = read_NNet(path)
    @assert controller_nnet == controller
end
```
