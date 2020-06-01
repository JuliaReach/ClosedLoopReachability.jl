using Documenter, NeuralNetworkAnalysis

DocMeta.setdocmeta!(NeuralNetworkAnalysis, :DocTestSetup,
                    :(using NeuralNetworkAnalysis); recursive=true)

makedocs(
    sitename = "NeuralNetworkAnalysis.jl",
    modules = [NeuralNetworkAnalysis],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = ["assets/juliareach.css"]),
    pages = [
        "Home" => "index.md",
        "About" => "about.md"
    ],
    strict = true
)

deploydocs(
    repo = "github.com/JuliaReach/NeuralNetworkAnalysis.jl.git",
)