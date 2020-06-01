using Documenter, NeuralNetworkAnalysis

DocMeta.setdocmeta!(NeuralNetworkAnalysis, :DocTestSetup,
                    :(using NeuralNetworkAnalysis); recursive=true)

makedocs(
    sitename = "NeuralNetworkAnalysis.jl",
    modules = [NeuralNetworkAnalysis],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "About" => "about.md"
    ],
    strict = false
)

deploydocs(
    repo = "github.com/JuliaReach/NeuralNetworkAnalysis.jl.git",
    push_preview=true
)
