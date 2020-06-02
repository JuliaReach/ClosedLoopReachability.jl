# generate examples
import Literate

MODELS = [
    joinpath(@__DIR__, "..", "models", "Sherlock-Benchmark-9"),
    joinpath(@__DIR__, "..", "models", "Sherlock-Benchmark-10")
]
GENERATEDDIR = joinpath(@__DIR__, "src", "models")
mkpath(GENERATEDDIR)
for model in MODELS
    for file in model
        if endswith(file, ".jl")
            input = abspath(joinpath(EXAMPLEDIR, model))
            script = Literate.script(input, GENERATEDDIR)
            code = strip(read(script, String))
            mdpost(str) = replace(str, "@__CODE__" => code)
            Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
            Literate.notebook(input, GENERATEDDIR, execute = true)
        elseif any(endswith.(file, [".png", ".jpg", ".gif"]))
            cp(joinpath(model, file), joinpath(GENERATEDDIR, file); force=true)
        else
            @warn "ignoring $file in $model"
        end
    end
end
