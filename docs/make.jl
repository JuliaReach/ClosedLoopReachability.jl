using Documenter, ClosedLoopReachability

DocMeta.setdocmeta!(ClosedLoopReachability, :DocTestSetup,
                    :(using ClosedLoopReachability); recursive=true)

# generate models
include("generate.jl")

makedocs(
    sitename = "ClosedLoopReachability.jl",
    modules = [ClosedLoopReachability],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             collapselevel = 1,
                             assets = ["assets/aligned.css"]),
    pages = [
        "Home" => "index.md",
        "Examples" => Any["Adaptive Cruise Controller (ACC)" => "models/ACC.md",
                          "Translational Oscillations (TORA) with ReLU controller" => "models/Sherlock-Benchmark-9-TORA.md",
                          "Translational Oscillations (TORA) with sigmoid controller" => "models/Sherlock-Benchmark-9-TORA-Sigmoid.md",
                          "Translational Oscillations (TORA) with ReLU/tanh controller" => "models/Sherlock-Benchmark-9-TORA-ReluTanh.md",
                          "Unicycle Car Model" => "models/Sherlock-Benchmark-10-Unicycle.md",
                          "Vertical Collision Avoidance (VCAS)" => "models/VertCAS.md",
                          "Single Inverted Pendulum" => "models/Single-Pendulum.md",
                          "Double Inverted Pendulum" => "models/Double-Pendulum.md",
                          "Airplane" => "models/Airplane.md"],
                          #"Sliding controller" => "models/Sherlock-Benchmark-7.md",
                          #"Nonlinear Cart-Pole" => "models/Cart-Pole.md"],
        "API Reference" => Any["Problem types"=>"lib/problems.md",
                               "Solvers"=>"lib/solvers.md",
                               "Utilities"=>"lib/utils.md"]
    ],
    strict = false
)

deploydocs(
    repo = "github.com/JuliaReach/ClosedLoopReachability.jl.git",
    push_preview = true
)
