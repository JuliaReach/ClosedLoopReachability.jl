abstract type AbstractNetwork end

struct Network <: AbstractNetwork
    layers::Vector{Layer}
end

function Base.:(==)(N1::Network, N2::Network)
    return N1.layers == N2.layers
end
