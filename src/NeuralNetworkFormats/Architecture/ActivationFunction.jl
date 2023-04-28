"""
    ActivationFunction

Abstract type for activation functions.
"""
abstract type ActivationFunction end


"""
    Id

Identity activation.

```math
    f(x) = x
```
"""
struct Id <: ActivationFunction end

(::Id)(x) = x


"""
    ReLU

Rectified linear unit (ReLU) activation.

```math
    f(x) = max(x, 0)
```
"""
struct ReLU <: ActivationFunction end

(::ReLU)(x) = max.(x, zero(eltype(x)))


"""
    Sigmoid

Sigmoid activation.

```math
    f(x) = \\frac{1}{1 + e^{-x}}
```
"""
struct Sigmoid <: ActivationFunction end

(::Sigmoid)(x) = @. 1 / (1 + exp(-x))


"""
    Tanh

Hyperbolic tangent activation.

```math
    f(x) = tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}
```
"""
struct Tanh <: ActivationFunction end

(::Tanh)(x) = tanh.(x)
