# DNN.jl - A light-weight library for representing deep neural networks

This light-weight library contains basic representations of deep neural networks
as well as functionality to parse them from various file formats such as MAT,
YAML, and ONNX.

This library originated from the package
[ClosedLoopReachability](https://github.com/JuliaReach/ClosedLoopReachability.jl),
which performs formal analysis of a given trained neural network.
This motivates that `DNN.jl` does not provide support for typical other tasks
such as network training, and some of the supported file formats are only used
by some similar analysis tool.

## Related packages

- [Flux.jl](https://github.com/FluxML/Flux.jl/) is a comprehensive Julia
  framework for machine learning. It also offers a representation of neural
  networks.
- [NNet](https://github.com/sisl/NNet) offers a representation of neural
  networks and a parser for the NNet format.
