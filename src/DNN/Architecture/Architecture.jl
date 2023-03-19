@reexport module Architecture

export AbstractNeuralNetwork, AbstractLayerOp,
       FeedforwardNetwork, DenseLayerOp,
       dim_in, dim_out, dim,
       ActivationFunction, Id, ReLU, Sigmoid, Tanh

include("AbstractNeuralNetwork.jl")
include("AbstractLayerOp.jl")
include("ActivationFunction.jl")
include("DenseLayerOp.jl")
include("FeedforwardNetwork.jl")

end  # module
