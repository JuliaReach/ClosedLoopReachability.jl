using Documenter, NeuralNetworkAnalysis

DocMeta.setdocmeta!(NeuralNetworkAnalysis, :DocTestSetup,
                    :(using NeuralNetworkAnalysis); recursive=true)

# Generate models
include("generate.jl")

makedocs(
    sitename = "NeuralNetworkAnalysis.jl",
    modules = [NeuralNetworkAnalysis],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Examples" => Any["3D model with sliding controller" => "models/Sherlock-Benchmark-7.md",
                          "TORA" => "models/Sherlock-Benchmark-9.md",
                          "Unicycle Car Model" => "models/Sherlock-Benchmark-10.md",
                          "Non-Linear Cart-Pole" => "models/Cart-Pole.md",
                          "Adaptive Cruise Controller" => "models/ACC.md",
#                           "Airplane" => "models/Airplane.md",
                          "Single Pendulum" => "models/SinglePendulum.md",
                          "Double Pendulum" => "models/DoublePendulum.md",
                          "Vertical CAS" => "models/VertCAS.md",
                         ],
        "About" => "about.md"
    ],
    strict = false
)

deploydocs(
    repo = "github.com/JuliaReach/NeuralNetworkAnalysis.jl.git",
    push_preview=true
)
