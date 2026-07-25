ENV["GKSwstype"] = "100"  # prevent plots from opening interactively

using Documenter, ClosedLoopReachability, DocumenterCitations
import Plots

DocMeta.setdocmeta!(ClosedLoopReachability, :DocTestSetup,
                    :(using ClosedLoopReachability); recursive=true)

# generate models
include("generate.jl")

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:alpha)

makedocs(; sitename="ClosedLoopReachability.jl",
         modules=[ClosedLoopReachability],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true",
                                collapselevel=1,
                                assets=["assets/aligned.css", "assets/citations.css"]),
         pagesonly=true,
         plugins=[bib],
         pages=["Home" => "index.md",
                "Examples" => Any[
                                  #
                                  "Adaptive Cruise Control (ACC)" => "models/ACC.md",
                                  "Airplane" => "models/Airplane.md",
                                  "Attitude Control" => "models/AttitudeControl.md",
                                  "Inverted Pendulum" => "models/InvertedPendulum.md",
                                  "Inverted Two-Link Pendulum" => "models/InvertedTwoLinkPendulum.md",
                                  "Quadrotor" => "models/Quadrotor.md",
                                  "Spacecraft Docking" => "models/SpacecraftDocking.md",
                                  "Translational Oscillations by a Rotational Actuator (TORA)" => "models/TORA.md",
                                  "Unicycle" => "models/Unicycle.md",
                                  "Vertical Collision Avoidance System (VerticalCAS)" => "models/VerticalCAS.md"
                                  #
                                  ],
                "API Reference" => Any["Problem types" => "lib/problems.md",
                                       "Solvers" => "lib/solvers.md"],
                "Bibliography" => "bibliography.md"])

deploydocs(; repo="github.com/JuliaReach/ClosedLoopReachability.jl.git",
           push_preview=true)
