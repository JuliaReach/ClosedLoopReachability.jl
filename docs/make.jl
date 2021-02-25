using Documenter, NeuralNetworkAnalysis

DocMeta.setdocmeta!(NeuralNetworkAnalysis, :DocTestSetup,
                    :(using NeuralNetworkAnalysis); recursive=true)

# Generate models
include("generate.jl")

makedocs(
    sitename = "NeuralNetworkAnalysis.jl",
    modules = [NeuralNetworkAnalysis],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             collapselevel = 1,
                             assets = ["assets/juliareach-light.css"]),
    pages = [
        "Home" => "index.md",
        "Examples" => Any["Adaptive Cruise Controller (ACC)" => "models/ACC.md",
                          "Translational Oscillations (TORA)" => "models/Sherlock-Benchmark-9-TORA.md",
                          "Unicycle Car Model" => "models/Sherlock-Benchmark-10-Unicycle.md",
                          "Vertical Collision Avoidance (VCAS)" => "models/VertCAS.md",
                          "Single Inverted Pendulum" => "models/Single-Pendulum.md",
                          "Double Inverted Pendulum" => "models/Double-Pendulum.md",
                          "Airplane" => "models/Airplane.md",
                          "Sliding controller" => "models/Sherlock-Benchmark-7.md",
                          "Nonlinear Cart-Pole" => "models/Cart-Pole.md"],
        "API Reference" => Any["Problem types"=>"lib/problems.md",
                               "Solvers"=>"lib/solvers.md",
                               "Utilities"=>"lib/utils.md"],
        "About" => "about.md"
    ],
    strict = false
)

deploydocs(
    repo = "github.com/JuliaReach/NeuralNetworkAnalysis.jl.git",
    push_preview=true
)
