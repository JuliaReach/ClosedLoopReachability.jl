import Literate
import ClosedLoopReachability: @modelpath

MODELS = [
          joinpath(@__DIR__, "..", "models", "ACC"),
          joinpath(@__DIR__, "..", "models", "Airplane"),
          joinpath(@__DIR__, "..", "models", "AttitudeControl"),
          joinpath(@__DIR__, "..", "models", "InvertedPendulum"),
          joinpath(@__DIR__, "..", "models", "InvertedTwoLinkPendulum"),
          joinpath(@__DIR__, "..", "models", "Quadrotor"),
          joinpath(@__DIR__, "..", "models", "SpacecraftDocking"),
          joinpath(@__DIR__, "..", "models", "TORA"),
          joinpath(@__DIR__, "..", "models", "Unicycle"),
          joinpath(@__DIR__, "..", "models", "VerticalCAS")
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
            if get(ENV, "DOCUMENTATIONGENERATOR", "") == "true"
                Literate.markdown(input, GENERATEDDIR; postprocess=mdpost, credit=false)
            else
                # for the local build, one needs to set `nbviewer_root_url`
                Literate.markdown(input, GENERATEDDIR; postprocess=mdpost, credit=false, nbviewer_root_url="..")
            end
            Literate.notebook(input, GENERATEDDIR; execute=true, credit=false)
        elseif any(endswith.(file, [".png", ".jpg", ".gif"]))
            cp(joinpath(model, file), joinpath(GENERATEDDIR, file); force=true)
        elseif any(endswith.(file, [".md", ".polar"]))
            # ignore *.md files and controller files without warning
        else
            @warn "ignoring $file in $model"
        end
    end
end
