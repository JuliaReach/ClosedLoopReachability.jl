abstract type AbstractNeuralNetwork end

dim(N::AbstractNeuralNetwork) = (dim_in(N), dim_out(N))
