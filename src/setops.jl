using LazySets: _leq, _geq, isapproxzero

@inline _bounds(H, i) = (H.center[i] - H.radius[i], H.center[i] + H.radius[i])

relu(Z::Zonotope) = relu!(copy(Z))

# proof-of-concept implementation of Theorem 3.1 in
# Fast and Effective Robustness Certification,
# G. Singh, T. Gehr, M. Mirman, M. Püschel, M. Vechev
# https://files.sri.inf.ethz.ch/website/papers/DeepZ.pdf
function relu!(Z::Zonotope{N}) where {N}
    c = Z.center
    G = Z.generators
    n, m = size(G)
    H = overapproximate(Z, Hyperrectangle)
    Gnew = zeros(N, n, n)

    @inbounds for i in 1:n
        lx, ux = _bounds(H, i)
        if !_leq(lx, zero(N))
            nothing
        elseif _leq(ux, zero(N)) || isapproxzero(lx)
            c[i] = zero(N)
            G[i, :] = zeros(N, m)
        else
            λ = ux / (ux - lx)
            μ = - λ * lx / 2
            c[i] = c[i] * λ + μ
            G[i, :] = G[i, :] .* λ
            Gnew[i, i] = μ
        end
    end
    return Zonotope(c, hcat(G, Gnew))
end

function _forward_layer_zono(W::AbstractMatrix, b::AbstractVector, ::ReLU, X::Zonotope)
    Y = affine_map(W, Z, b)
    return relu(Y)
end

function _forward_network(network::Network, X::Zonotope)
    nlayers = length(network.layers) # TODO getter function ?
    Z = copy(X)
    @inbounds for i in 1:nlayers
        W = network.layers[i].weights
        b = network.layers[i].bias
        Z = _forward_layer_zono(W, b, activation, Z)
    end
    return remove_zero_generators(Z)
end
