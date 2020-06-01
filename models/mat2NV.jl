using NeuralVerification, MAT
using NeuralVerification: Layer, Network, ReLU, Id

_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
_vec(A::AbstractVector) = A

function readnn(file)
    vars = matread(file)
    if(first(keys(vars)) == "controller")
        dic = vars["controller"]
    else
        error("error, did not found key \"controller\"")
    end

    nLayers = Int(dic["number_of_layers"])
    layers = Vector{Layer}(undef, nLayers)
    aF = dic["activation_fcns"]

    for n = 1:nLayers
        W = dic["W"][n]
        b = dic["b"][n]
        if(aF[n] == "relu")
            act = ReLU()
        elseif(aF[n] == "linear")
            act = Id()
        else
            error("error, aF = $(aF[n]), nLayer = $n")
        end
        layers[n] = Layer(W, _vec(b), act)
    end

    return Network(layers)
end
