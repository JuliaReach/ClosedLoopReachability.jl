# 2D input vector and 2x3 layer
x = [1.0, 1]
W = hcat([1 0.5; -0.5 0.5; -1 -0.5])
b = [1.0, 0, -2]

# output for `x` under identity activation
L = DenseLayerOp(W, b, Id())
@test L(x) == W * x + b == [2.5, 0, -3.5]

# invalid weight/bias combination
@test_throws ArgumentError DenseLayerOp(W, [1.0, 0], Id())

# equality
@test L == DenseLayerOp(W, b, Id())
@test L != DenseLayerOp(W .+ 1, b, Id()) &&
    L != DenseLayerOp(W, b .+ 1, Id()) &&
    L != DenseLayerOp(W, b, ReLU())

# dimensions
@test dim_in(L) == 2
@test dim_out(L) == 3
@test dim(L) == (2, 3)

# test methods for all activations
function test_layer(L::DenseLayerOp{Id})
    @test L(x) == [2.5, 0, -3.5]
end

function test_layer(L::DenseLayerOp{ReLU})
    @test L(x) == [2.5, 0, 0]
end

function test_layer(L::DenseLayerOp{Sigmoid})
    @test L(x) ≈ [0.924, 0.5, 0.029]  atol=1e-3
end

function test_layer(L::DenseLayerOp{Tanh})
    @test L(x) ≈ [0.986, 0, -0.998]  atol=1e-3
end

function test_layer(L::DenseLayerOp)
    error("untested activation function: ", typeof(L.activation))
end

# run test with all activations
for act in subtypes(ActivationFunction)
    test_layer(DenseLayerOp(W, b, act()))
end
