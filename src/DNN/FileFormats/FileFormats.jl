"""
    FileFormats

Module to parse and write file formats of neural networks.
"""
module FileFormats

using ..Architecture
using Requires
using ReachabilityBase.Require

export read_MAT,
       read_NNet,
       read_ONNX,
       read_POLAR,
       read_Sherlock, write_Sherlock,
       read_YAML

include("init.jl")

include("available_activations.jl")

include("MAT.jl")
include("NNet.jl")
include("ONNX.jl")
include("POLAR.jl")
include("Sherlock.jl")
include("YAML.jl")

end  # module
