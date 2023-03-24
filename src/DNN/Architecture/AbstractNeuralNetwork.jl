"""
    AbstractNeuralNetwork

Abstract type for neural networks.
"""
abstract type AbstractNeuralNetwork end

"""
    dim_in(N::AbstractNeuralNetwork)

Return the input dimension of a neural network.

### Input

- `N` -- neural network

### Output

The dimension of the input layer of `N`.
"""
function dim_in(::AbstractNeuralNetwork) end

"""
    dim_out(N::AbstractNeuralNetwork)

Return the output dimension of a neural network.

### Input

- `N` -- neural network

### Output

The dimension of the output layer of `N`.
"""
function dim_out(::AbstractNeuralNetwork) end

"""
    dim(N::AbstractNeuralNetwork)

Return the input and output dimension of a neural network.

### Input

- `N` -- neural network

### Output

The pair ``(i, o)`` where ``i`` is the input dimension and ``o`` is the output
dimension of `N`.
"""
dim(N::AbstractNeuralNetwork) = (dim_in(N), dim_out(N))
