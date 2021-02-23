module NeuralNetworkAnalysis

include("init.jl")
include("problem.jl")
include("nnops.jl")
include("setops.jl")
include("utils.jl")
include("simulate.jl")
include("solve.jl")

export ControlledPlant,
       forward,
       simulate

end
