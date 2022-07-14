abstract type Solver end

# ================================================
# Internal forward network functions
# ================================================

# output of neural network for a single input
function forward(nnet::Network, x0::Vector{<:Number})
    x = x0
    @inbounds for layer in nnet.layers
        W = layer.weights
        b = layer.bias
        x = layer.activation(W * x + b)
    end
    return x
end

function forward_network(solver::Solver, nnet::Network, X0)
    X = X0
    for layer in nnet.layers
        _, X = forward_layer(solver, layer, X)
    end
    return X
end

function forward_layer(solver, layer, reach)
    Yl = forward_linear(solver, layer, reach)
    Y = forward_act(solver, layer, Yl)
    return Yl, Y
end

# ================================================
# Composite methods to compute the network output
# ================================================

@with_kw struct SplitSolver{S<:Solver, FS, FM} <: Solver
    solver::S
    split_fun::FS
    merge_fun::FM
end

function SplitSolver(solver)
    # default: box approximation and split in two sets per dimension
    split_fun = X -> split(box_approximation(X), 2 * ones(Int, dim(X)))
    # default: box approximation of the union
    merge_fun = X -> box_approximation(X)
    return SplitSolver(solver, split_fun, merge_fun)
end

function forward_network(solver::SplitSolver, nnet::Network, X0)
    X0_split = solver.split_fun(X0)
    Y_union = UnionSetArray()
    for X in X0_split
        Y = forward_network(solver.solver, nnet, X)
        push!(array(Y_union), Y)
    end
    Y_merged = solver.merge_fun(Y_union)
    return Y_merged
end

# ============================================================================
# Methods to approximate the network output (without mathematical guarantees)
# ============================================================================

# solver using the CH of the sampled outputs as an inner approx of the real output
@with_kw struct SampledApprox <: Solver
    nsamples::Int = 10000
    include_vertices::Bool = true
    directions = OctDirections
end

function forward_network(solver::SampledApprox, nnet, input)
    samples = sample(input, solver.nsamples;
                     include_vertices=solver.include_vertices)

    m = output_dim(nnet)
    if m == 1
        MIN = Inf
        MAX = -Inf
        for sample in samples
            output = first(forward(nnet, sample))
            MIN = min(MIN, output)
            MAX = max(MAX, output)
        end
        return Interval(MIN, MAX)
    else
        vlist = Vector{Vector{eltype(samples[1])}}(undef, length(samples))
        @inbounds for (i, sample) in enumerate(samples)
            vlist[i] = forward(nnet, sample)
        end
        convex_hull!(vlist)
        P = VPolytope(vlist)
        Z = overapproximate(P, Zonotope, solver.directions)
        return Z
    end
end

# ==============================================================================
# Method to handle networks with ReLU, sigmoid, and Tanh activation functions
# from [FAS18]
#
# [FAS18]: Singh, Gagandeep, et al. "Fast and Effective Robustness
# Certification." NeurIPS 2018.
# ==============================================================================

struct DeepZ <: Solver end

function forward_linear(solver::DeepZ, L::Layer, Z::AbstractZonotope)
    return affine_map(L.weights, Z, L.bias)
end

function forward_act(solver::DeepZ, L::Layer{Id}, Z::AbstractZonotope)
    return Z
end

function forward_act(solver::DeepZ, L::Layer{ReLU}, Z::AbstractZonotope)
    return overapproximate(Rectification(Z), Zonotope)  # implemented in LazySets
end

function sigmoid(x::Number)
    ex = exp(x)
    return ex / (1 + ex)
end

function sigmoid2(x::Number)
    ex = exp(x)
    return ex / (1 + ex)^2
end

function _overapproximate_zonotope(Z::AbstractZonotope{N}, act, act′) where {N}
    c = copy(center(Z))
    G = copy(genmat(Z))
    n, m = size(G)
    row_idx = Vector{Int}()
    μ_idx = Vector{N}()

    @inbounds for i in 1:n
        lx, ux = low(Z, i), high(Z, i)
        ly, uy = act(lx), act(ux)

        if LazySets._isapprox(lx, ux)
            c[i] = uy
            for j in 1:m
                G[i, j] = zero(N)
            end
        else
            λ = min(act′(lx), act′(ux))
            μ₁ = (uy + ly - λ * (ux + lx)) / 2
            # Note: there is a typo in the paper (missing parentheses)
            μ₂ = (uy - ly - λ * (ux - lx)) / 2
            c[i] = c[i] * λ + μ₁
            for j in 1:m
                G[i, j] = G[i, j] * λ
            end
            push!(row_idx, i)
            push!(μ_idx, μ₂)
        end
    end

    q = length(row_idx)
    if q >= 1
        Gnew = zeros(N, n, q)
        j = 1
        @inbounds for i in row_idx
            Gnew[i, j] = μ_idx[j]
            j += 1
        end
        Gout = hcat(G, Gnew)
    else
        Gout = G
    end

    return Zonotope(c, LazySets.remove_zero_columns(Gout))
end

function forward_act(solver::DeepZ, L::Layer{Sigmoid}, Z::AbstractZonotope)
    act(x) = sigmoid(x)
    act′(x) = sigmoid2(x)
    return _overapproximate_zonotope(Z, act, act′)
end

function forward_act(solver::DeepZ, L::Layer{Tanh}, Z::AbstractZonotope)
    act(x) = tanh(x)
    act′(x) = 1 - tanh(x)^2
    return _overapproximate_zonotope(Z, act, act′)
end

# ==========================================================
# Methods to handle networks with ReLU activation functions
# ==========================================================

# solver that computes the box approximation in each layer
# it exploits that box(relu(X)) == relu(box(X))
struct BoxSolver <: Solver end

function forward_network(solver::BoxSolver, nnet::Network, X0)
    X = X0
    for layer in nnet.layers
        # affine map and box approximation
        W = layer.weights
        b = layer.bias
        X_am = AffineMap(W, X, b)

        # activation function
        if layer.activation isa Id
            X = X_am
            continue
        end

        @assert layer.activation isa ReLU "unsupported activation function"
        X = rectify(box_approximation(X_am))
    end
    return X
end

@with_kw struct ConcreteReLU <: Solver
    concrete_intersection::Bool = false
    convexify::Bool = false
end

function forward_network(solver::ConcreteReLU, nnet::Network, X0)
    X = [X0]
    for layer in nnet.layers
        if typeof(X[1]) <: UnionSetArray
            X = [x.array for x in X]
            X = reduce(vcat, X)
        end
        X = affine_map.(Ref(layer.weights), X, Ref(layer.bias))

        # activation function
        if layer.activation isa Id
            continue
        end
        @assert layer.activation isa ReLU "unsupported activation function"
        X = rectify.(X, solver.concrete_intersection)
    end
    return solver.convexify ? ConvexHullArray(X) : X
end

# solver that propagates the vertices, computes their convex hull, and applies
# some postprocessing to the result
@with_kw struct VertexSolver{T} <: Solver
    postprocessing::T = x -> x  # default: identity (= no postprocessing)
    apply_convex_hull::Bool = false
end

function forward_network(solver::VertexSolver, nnet::Network, X0)
    N = eltype(X0)
    P = X0

    for layer in nnet.layers
        # apply affine map
        W = layer.weights
        b = layer.bias
        P = convert(VPolytope, P)
        Q_am = affine_map(W, P, b)

        # activation function
        if layer.activation isa Id
            P = Q_am
            continue
        end

        @assert layer.activation isa ReLU "unsupported activation function"

        # compute Q_chull = convex_hull(Q_am, rectify(Q_am))
        vlist = Vector{Vector{N}}()
        vlist_rect = Vector{Vector{N}}()
        for v in vertices(Q_am)
            push!(vlist, v)
            v_rect = rectify(v)
            push!(vlist_rect, v_rect)
            if v != v_rect
                push!(vlist, v_rect)
            end
        end
        if solver.apply_convex_hull || true
            convex_hull!(vlist)
        end
        Q_chull = VPolytope(vlist)

        # filter out negative part
#         Q_pos = box_approximation(VPolytope(vlist_rect))  # alternative
        n = dim(Q_am)
        Q_pos = HPolyhedron(
            [HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)) for i in 1:n])
        P = intersection(Q_chull, Q_pos)
    end
    Q = solver.postprocessing(P)
    return Q
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
                 alg=TMJets(abstol=1e-14, orderQ=2, orderT=6))

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

# =====================================
# Methods to handle arbitrary networks
# =====================================

struct BlackBoxSolver <: Solver end

struct BlackBoxController{FT} <: AbstractNetwork
    f::FT
end

function forward_network(solver::BlackBoxSolver, bbc::BlackBoxController, X0)
    return bbc.f(X0)
end

function forward(bbc::BlackBoxController, X0)
    return bbc.f(X0)
end

# =============================
# Handling of singleton inputs
# =============================

function forward(nnet::Network, X0::AbstractSingleton)
    x0 = element(X0)
    x1 = forward(nnet, x0)
    return Singleton(x1)
end

for SOLVER in LazySets.subtypes(Solver, true)
    @eval function forward_network(solver::$SOLVER, nnet::Network,
                                   X0::AbstractSingleton)
              return forward(nnet, X0)
          end
end
