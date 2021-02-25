# # Translational Oscillations by a Rotational Actuator (TORA)
#
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`Sherlock-Benchmark-9.ipynb`](__NBVIEWER_ROOT_URL__/../Sherlock-Benchmark-9-TORA.ipynb)
#
#
# This model consists of a cart attached to a wall with a spring, and is free
# to move a friction-less surface. The car itself has a weight attached to an arm
# inside it, which is free to rotate about an axis as described in the figure below.
# This serves as the control input, in order to stabilize the cart at ``x = 0``.
#
# ## Model
#
# The model is four dimensional, given by the following equations:
#
# ```math
# \begin{aligned}
# \dot{x}_1 &= x_2 \\
# \dot{x}_2 &= -x_1 + 0.1 \sin x_3 \\
# \dot{x}_3 &= x_4  \\
# \dot{x}_4 &= u
# \end{aligned}
# ```

using NeuralNetworkAnalysis

@taylorize function benchmark9!(dx, x, p, t)
    u = one(x[4])
    aux = 0.1 * sin(x[3])
    dx[1] = x[2]
    dx[2] = -x[1] + aux
    dx[3] = x[4]
    dx[4] = u
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])

prob = @ivp(x' = benchmark9!(x), dim: 4, x(0) ∈ X₀)

# TODO add normalization of control inputs to u = NN(x) - 10

# solve it
##sol = solve(prob, T=20.0);

# ## Specifications
#
# The verification problem here is that of safety. For an initial set of
# ``x_1 ∈ [0.6, 0.7], x_2 ∈ [−0.7,−0.6], x_3 ∈ [−0.4,−0.3], x_4 ∈ [0.5,0.6]``,
# it is required to prove that thesystem stays within the box ``x ∈ [−1, 1]^3``,
# for a time window of 20s.

# ## Results

# ## References

#
#
