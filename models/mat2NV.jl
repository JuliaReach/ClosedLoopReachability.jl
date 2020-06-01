using NeuralVerification, MAT
using NeuralVerification: Layer, Network, ReLU, Id

_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
_vec(A::AbstractVector) = A


"""
    readnn(file, key="controller")

Return the list of vertices of a polyhedron in constraint representation.

### Input

- `file` -- string indicating the location of the file containing the NN
- `key` -- (optional, default: `"controller"`) key used to search the dictionary

### Output

A Neural Network using the internal format of NeuralVerification.jl

"""
function readnn(file, key="controller")
    vars = matread(file)
    !haskey(vars, key) && throw(ArgumentError("didn't find key $key"))
    dic = vars[key]

    nLayers = Int(dic["number_of_layers"])
    layers = Vector{Layer}(undef, nLayers)
    aF = dic["activation_fcns"]

    for n = 1:nLayers
        W = dic["W"][n]
        b = dic["b"][n]
        if aF[n] == "relu"
            act = ReLU()
        elseif aF[n] == "linear"
            act = Id()
        else
            error("error, aF = $(aF[n]), nLayer = $n")
        end
        layers[n] = Layer(W, _vec(b), act)
    end

    return Network(layers)
end
