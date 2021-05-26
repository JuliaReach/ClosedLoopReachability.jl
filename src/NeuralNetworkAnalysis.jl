module NeuralNetworkAnalysis

include("init.jl")
include("problem.jl")
include("nnops.jl")
include("setops.jl")
include("split.jl")
include("utils.jl")
include("simulate.jl")
include("solve.jl")

# problem types
export ControlledPlant

# splitters
export BoxSplitter, ZonotopeSplitter

# solvers
export solve, forward, simulate,
       SampledApprox, VertexSolver, BoxSolver, SplitSolver

# utility functions
export @modelpath, read_nnet_mat, read_nnet_yaml

end
