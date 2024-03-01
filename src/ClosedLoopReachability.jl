module ClosedLoopReachability

include("init.jl")
include("problem.jl")
include("setops.jl")
include("splitters.jl")
include("utils.jl")
include("simulate.jl")
include("solve.jl")

# problem types
export ControlledPlant

# splitters
export BoxSplitter,
       ZonotopeSplitter,
       IndexedSplitter,
       SignSplitter

# solvers
export solve, simulate

end  # module
