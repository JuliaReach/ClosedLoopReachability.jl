struct DenseLayerOp{F, M<:AbstractMatrix, B} <: AbstractLayerOp
    weights::M
    bias::B
    activation::F
end

(l::DenseLayerOp)(x) = l.activation.(l.weights * x .+ l.bias)

Base.length(L::DenseLayerOp) = length(L.bias)

function Base.:(==)(L1::DenseLayerOp, L2::DenseLayerOp)
    return L1.weights == L2.weights &&
           L1.bias == L2.bias &&
           L1.activation == L2.activation
end
