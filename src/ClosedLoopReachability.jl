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

# solve
export solve

# simulation
export simulate,
       trajectory,
       trajectories,
       controls,
       disturbances,
       solutions

# plotting
export plot_simulation!

end  # module
