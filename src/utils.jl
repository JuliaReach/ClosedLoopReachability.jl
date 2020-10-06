using NeuralVerification, MAT
using NeuralVerification: Layer, Network, ReLU, Id

_vec(A::AbstractMatrix) = vec(A)
_vec(A::Number) = [A]
_vec(A::AbstractVector) = A


"""
    readnn(file; key="controller")

Return the neural network using the internal format of NeuralVerification.jl.

### Input

- `file` -- string indicating the location of the file containing the neural network
- `key` -- (optional, default: `"controller"`) key used to search the dictionary

### Output

A Neural Network using the internal format of NeuralVerification.jl

"""
function readnn(file; key="controller")
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

function scalar_quadratic!(du, u, p, t)
    xᴶ, xᴾ = u
    du[1] = zero(xᴶ)
    du[2] = xᴶ *(xᴾ - xᴾ^2)
end

function dnn2hybrid(nn::Network, xᴾ₀, u₀)
    I1 = IntervalArithmetic.Interval(1., 1.)
    xᴾ = [I1, I1] * 0.5
    h = []
    for layer in nn.layers
        n = length(layer.bias)
        xᴶ = layer.weights * xᴾ₀ + layer.bias
        if layer.activation == NV.Sigmoid()
            for i=1:n
                X0 = xᴾ[i] × xᴶ[i]
                ivp = @ivp(x' = scalar_quadratic!(x), dim=2, x(0) ∈ X0)
                sol = RA.solve(ivp, tspan=(0., 1.),
                             alg=TMJets(abs_tol=1e-14, orderQ=2, orderT=6));
                push!(h, sol.F.ext[:xv][end][2])
            end
        else
            push!(h, xᴶ)
        end
        println(h)
        xᴾ = h
        h = []
    end
    return xᴾ
end
