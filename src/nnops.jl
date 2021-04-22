using NeuralVerification: @with_kw,
                          ActivationFunction,
                          Solver,
                          Sigmoid,
                          Tanh

# ================================================
# Internal forward network functions
# ================================================

# output of neural network for a single input
function forward(network::Network, x0::Vector{<:Number})
    layers = network.layers
    x = x0
    @inbounds for i in 1:length(layers)
        layer = network.layers[i]
        W = layer.weights
        b = layer.bias
        x = layer.activation(W * x + b)
    end
    return x
end

# ==========================================================
# Methods to handle networks with ReLU activation functions
# ==========================================================

# solver that propagates the vertices, computes their convex hull, and applies
# some postprocessing to the result
@with_kw struct VertexSolver{T} <: Solver
    postprocessing::T = x -> x  # default: identity (= no postprocessing)
end

function NeuralVerification.forward_network(solver::VertexSolver, nnet::Network, X0)
    P = VPolytope()
    vlist = vertices_list(P)
    for v in vertices(X0)
        push!(vlist, forward(nnet, v))
    end
    Q = solver.postprocessing(P)
    return Q
end

# solver using the CH of the sampled outputs as an inner approx of the real output
@with_kw struct SampledApprox <: Solver
    nsamples::Int = 10000
end

function NeuralVerification.forward_network(solver::SampledApprox, nnet, input)
    @assert output_dim(nnet) == 1 "the dimension of the output of the network needs to be 1, but is $output_dim(nnet)"
    samples = sample(input, solver.nsamples)
    MIN = Inf
    MAX = -Inf
    for sample in samples
        output = first(NV.compute_output(nnet, sample))
        MIN = min(MIN, output)
        MAX = max(MAX, output)
    end
    return Interval(MIN, MAX)
end

@with_kw struct ConcreteReLU <: Solver
    concrete_intersection::Bool = false
end

function NeuralVerification.forward_network(solver::ConcreteReLU, nnet::Network, X0)
    X = [X0]
    for layer in nnet.layers
        if typeof(X[1]) <: UnionSetArray
            X = [x.array for x in X]
            X = reduce(vcat, X)
        end
        X = affine_map.(Ref(layer.weights), X, Ref(layer.bias))
        X = rectify.(X, solver.concrete_intersection)
    end
    return X
end

# ==============================================================================
# Methods to handle networks with sigmoid activation functions from [VER19]
#
# [VER19]: Ivanov, Radoslav, et al. "Verisig: verifying safety properties of
# hybrid systems with neural network controllers." Proceedings of the 22nd ACM
# International Conference on Hybrid Systems: Computation and Control. 2019.
# ==============================================================================

# ref. Eq (6) in [VER19]
# d(σ(x))/dx = σ(x)*(1-σ(x))
# g(t, x) = σ(tx) = 1 / (1 + exp(-tx))
# dg(t, x)/dt = g'(t, x) = x * g(t, x) * (1 - g(t, x))
@taylorize function sigmoid!(dx, x, p, t)
    xᴶ, xᴾ = x
    dx[1] = zero(xᴶ)
    dx[2] = xᴶ *(xᴾ - xᴾ^2)
end

# footnote (3) in [VER19]
# d(tanh(x))/dx = 1 - tanh(x)^2
# g(t, x) = tanh(tx)
# dg(t, x)/dt = g'(t, x) = x * (1 - g(t, x)^2)
@taylorize function tanh!(dx, x, p, t)
    xᴶ, xᴾ = x
    dx[1] = zero(xᴶ)
    dx[2] = xᴶ *(1 - xᴾ^2)
end

const HALFINT = IA.Interval(0.5, 0.5)
const ZEROINT = IA.Interval(0.0, 0.0)
const ACTFUN = Dict(Tanh() => (tanh!, ZEROINT),
                    Sigmoid() => (sigmoid!, HALFINT))

# Method: Cartesian decomposition (intervals for each one-dimensional subspace)
# Only Tanh, Sigmoid and Id functions are supported
function forward(nnet::Network, X0::LazySet;
                 alg=TMJets(abs_tol=1e-14, orderQ=2, orderT=6))

    # initial states
    xᴾ₀ = _decompose_1D(X0)
    xᴾ₀ = LazySets.array(xᴾ₀)  # see https://github.com/JuliaReach/ReachabilityAnalysis.jl/issues/254
    xᴾ₀ = [x.dat for x in xᴾ₀] # use concrete inteval matrix-vector operations

    for layer in nnet.layers  # loop over layers
        W = layer.weights
        m, n = size(W)
        b = layer.bias
        act = layer.activation

        xᴶ′ = W * xᴾ₀ + b  # (scalar matrix) * (interval vector) + (scalar vector)

        if act == Id()
            xᴾ₀ = copy(xᴶ′)
            continue
        end

        activation!, ival = ACTFUN[act]
        xᴾ′ = fill(ival, m)

        for i = 1:m  # loop over coordinates
            X0i = xᴶ′[i] × xᴾ′[i]
            ivp = @ivp(x' = activation!(x), dim=2, x(0) ∈ X0i)
            sol = RA.solve(ivp, tspan=(0., 1.), alg=alg)

            # interval overapproximation of the final reach-set along
            # dimension 2, which corresponds to xᴾ
            xᴾ_end = sol.F.ext[:xv][end][2]
            xᴾ′[i] = xᴾ_end
        end
        xᴾ₀ = copy(xᴾ′)
    end
    return CartesianProductArray([Interval(x) for x in xᴾ₀])
end

function apply(normalization::UniformAdditiveNormalization, X::LazySet)
    return translate(X, fill(normalization.shift, dim(X)))
end
