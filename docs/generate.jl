import Literate
import ReachabilityBase.CurrentPath: @current_path

MODELS = [
          #
          "ACC",
          "Airplane",
          "AttitudeControl",
          "InvertedPendulum",
          "InvertedTwoLinkPendulum",
          "Quadrotor",
          "SpacecraftDocking",
          "TORA",
          "Unicycle",
          "VerticalCAS"
          #
          ]

source_dir = joinpath(@__DIR__, "..", "models")
target_dir = joinpath(@__DIR__, "src", "models")
mkpath(target_dir)

# overwrite to use the correct model path
macro current_path(prefix::String, filename::String)
    return joinpath(source_dir, prefix, filename)
end

for model in MODELS
    model_path = abspath(joinpath(source_dir, model))
    for file in readdir(model_path)
        if endswith(file, ".jl")
            input = abspath(joinpath(model_path, file))
            script = Literate.script(input, target_dir; credit=false)
            code = strip(read(script, String))
            mdpost(str) = replace(str, "@__CODE__" => code)
            if get(ENV, "DOCUMENTATIONGENERATOR", "") == "true"
                Literate.markdown(input, target_dir; postprocess=mdpost, credit=false)
            else
                # for the local build, one needs to set `nbviewer_root_url`
                Literate.markdown(input, target_dir; postprocess=mdpost, credit=false,
                                  nbviewer_root_url="..")
            end

            # notebooks are deactivated to speed up the generation
            # Literate.notebook(input, target_dir_md; execute=true, credit=false)
            # if used, add the following to the top of the script files (where `MODELNAME` is the model name):
            #md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/MODELNAME.ipynb)
        elseif any(endswith.(file, [".png", ".jpg", ".gif"]))
            cp(joinpath(model_path, file), joinpath(target_dir, file); force=true)
        elseif any(endswith.(file, [".md", ".polar"]))
            # ignore *.md files and controller files without warning
        else
            @warn "ignoring $file in $model_path"
        end
    end
end
