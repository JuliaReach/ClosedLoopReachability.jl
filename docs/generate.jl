# generate examples
import Literate
import ClosedLoopReachability: @modelpath

MODELS = [joinpath(@__DIR__, "..", "models", "ACC"),
          joinpath(@__DIR__, "..", "models", "Sherlock-Benchmark-9-TORA"),
          joinpath(@__DIR__, "..", "models", "Sherlock-Benchmark-10-Unicycle"),
          joinpath(@__DIR__, "..", "models", "VertCAS"),
          joinpath(@__DIR__, "..", "models", "Single-Pendulum"),
          joinpath(@__DIR__, "..", "models", "Double-Pendulum"),
          joinpath(@__DIR__, "..", "models", "Airplane"),
          joinpath(@__DIR__, "..", "models", "AttitudeControl"),
          joinpath(@__DIR__, "..", "models", "Quadrotor"),
          joinpath(@__DIR__, "..", "models", "Spacecraft")
          #joinpath(@__DIR__, "..", "models", "Sherlock-Benchmark-7"),
          #joinpath(@__DIR__, "..", "models", "Cart-Pole"),
          ]
GENERATEDDIR = joinpath(@__DIR__, "src", "models")
MODELDIR = joinpath(@__DIR__, "..", "models")
mkpath(GENERATEDDIR)

macro modelpath(model_path::String, name::String)
    return joinpath(MODELDIR, model_path, name)
end

for model in MODELS
    for file in readdir(model)
        if endswith(file, ".jl")
            input = abspath(joinpath(model, file))
            script = Literate.script(input, GENERATEDDIR; credit=false)
            code = strip(read(script, String))
            mdpost(str) = replace(str, "@__CODE__" => code)
            Literate.markdown(input, GENERATEDDIR; postprocess=mdpost, credit=false)
            Literate.notebook(input, GENERATEDDIR; execute=true, credit=false)
        elseif any(endswith.(file, [".png", ".jpg", ".gif"]))
            cp(joinpath(model, file), joinpath(GENERATEDDIR, file); force=true)
        else
            @warn "ignoring $file in $model"
        end
    end
end
