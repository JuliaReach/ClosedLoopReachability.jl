abstract type ActivationFunction end


struct Id <: ActivationFunction end

(::Id)(x) = x


struct ReLU <: ActivationFunction end

(::ReLU)(x) = max.(x, zero(eltype(x)))


struct Sigmoid <: ActivationFunction end

(::Sigmoid)(x) = @. 1 / (1 + exp(-x))


struct Tanh <: ActivationFunction end

(::Tanh)(x) = tanh.(x)
