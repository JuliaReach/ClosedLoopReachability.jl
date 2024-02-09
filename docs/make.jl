using Documenter, ClosedLoopReachability

DocMeta.setdocmeta!(ClosedLoopReachability, :DocTestSetup,
                    :(using ClosedLoopReachability); recursive=true)

# generate models
include("generate.jl")

makedocs(; sitename="ClosedLoopReachability.jl",
         modules=[ClosedLoopReachability],
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true",
                                collapselevel=1,
                                assets=["assets/aligned.css"]),
         pagesonly=true,
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
                                       "Solvers" => "lib/solvers.md",
                                       "Utilities" => "lib/utils.md"]])

deploydocs(; repo="github.com/JuliaReach/ClosedLoopReachability.jl.git",
           push_preview=true)
