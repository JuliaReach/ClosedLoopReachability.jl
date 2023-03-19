# 2D input vector
x = [1.0, 1]

# 2x3 layer
W1 = hcat([1 0.5; -0.5 0.5; -1 -0.5])
b1 = [1.0, 0, -2]

# network with a single layer
L1 = DenseLayerOp(W1, b1, ReLU())
N1 = FeedforwardNetwork([L1])
@test N1(x) == max.(W1 * x + b1, 0) == [2.5, 0, 0]

# invalid layer combination
@test_throws ArgumentError FeedforwardNetwork([L1, L1])

# 3x2 layer
W2 = hcat([-1 -0.5 0; 0.5 -0.5 0])
b2 = [-1.0, 0]

# network with two layers
L2 = DenseLayerOp(W2, b2, Id())
N2 = FeedforwardNetwork([L1, L2])
@test N2(x) == W2 * max.(W1 * x + b1, 0) + b2 == [-3.5, 1.25]

# equality
@test N1 == FeedforwardNetwork([L1])
@test N1 != FeedforwardNetwork([L2])

# dimensions
@test dim_in(N1) == 2 && dim_in(N2) == 2
@test dim_out(N1) == 3 && dim_out(N2) == 2
@test dim(N1) == (2, 3) && dim(N2) == (2, 2)
