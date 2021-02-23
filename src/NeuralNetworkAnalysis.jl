module NeuralNetworkAnalysis

include("init.jl")
include("problem.jl")
include("nnops.jl")
include("setops.jl")
include("utils.jl")
include("solve.jl")

export ControlledPlant,
       forward

end
