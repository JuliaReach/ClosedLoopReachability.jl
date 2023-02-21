module DNN

using Requires

using ReachabilityBase.Require

export AbstractNeuralNetwork, Network, Layer,
       ActivationFunction, Id, ReLU, Sigmoid, Tanh,
       read_nnet,
       read_nnet_mat,
       read_nnet_yaml,
       read_nnet_sherlock, write_nnet_sherlock,
       read_nnet_onnx,
       read_nnet_polar

include("init.jl")

include("AbstractNeuralNetwork.jl")
include("ActivationFunction.jl")
include("Layer.jl")
include("Network.jl")

include("FileFormats/nnet.jl")
include("FileFormats/MAT.jl")
include("FileFormats/YAML.jl")
include("FileFormats/Sherlock.jl")
include("FileFormats/ONNX.jl")
include("FileFormats/POLAR.jl")

end  # module
