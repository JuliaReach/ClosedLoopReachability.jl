```@meta
DocTestSetup = :(using ClosedLoopReachability)
CurrentModule = ClosedLoopReachability
```

# DNN.jl (neural-network library)

```@docs
DNN
```

## Architecture.jl (DNN data structures)

```@docs
Architecture
```

### General interface for neural networks

```@docs
AbstractNeuralNetwork
dim_in(::AbstractNeuralNetwork)
dim_out(::AbstractNeuralNetwork)
ClosedLoopReachability.DNN.dim(::AbstractNeuralNetwork)
```

#### Implementation

```@docs
FeedforwardNetwork
```

### General interface for layer operations

```@docs
AbstractLayerOp
dim_in(::AbstractLayerOp)
dim_out(::AbstractLayerOp)
ClosedLoopReachability.DNN.dim(::AbstractLayerOp)
```

#### Implementation

```@docs
DenseLayerOp
```

### Activation functions

```@docs
ActivationFunction
Id
ReLU
Sigmoid
Tanh
```

The following strings can be parsed as activation functions:

```@example
using ClosedLoopReachability  # hide
ClosedLoopReachability.DNN.FileFormats.available_activations
```

## FileFormats.jl (DNN file formats)

```@docs
FileFormats
```

### Reading neural networks

```@docs
read_MAT
read_NNet
read_ONNX
read_POLAR
read_Sherlock
read_YAML
```

### Writing neural networks

```@docs
write_Sherlock
```