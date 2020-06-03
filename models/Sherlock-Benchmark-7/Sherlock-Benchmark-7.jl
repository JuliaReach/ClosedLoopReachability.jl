# # 3D system with sliding mode controller
#
# The model was first proposed in [1], where the authors originally designed a
# discontinuous sliding mode controller for this system.

# ## Model
#
# The model is a three-dimensional system given by the following equations:
#
# ```math
# \begin{aligned}
# \dot{x}_1 &= x_3^2 - x_2 + w \\
# \dot{x}_2 &= x_3 \\
# \dot{x}_3 &= u
# \end{aligned}
# ```
# where w is a bounded error in the range ``[−0.01, 0.01]``. A neural network controller
# was trained for this system, using a model predictive controller as a demonstrator.
# The trained network had 2 hidden layers, with 300 neurons in the first layer,
# and 200 in the second layer. The sampling time for this controller was 0.5s.

using NeuralNetworkAnalysis, Plots

# The spatial variables are aguemte
@taylorize function f!(dx, x, p, t)
    x₁, x₂, x₃, x₄, w, u = x

    dx[1] = x₃^3 - x₂ + w
    dx[2] = x₃
    dx[3] = u
    dx[4] = zero(x₄) # w
    dx[5] = zero(x₅) # u
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[0.35, 0.45, 0.25], high=[0.45, 0.55, 0.35])
W₀ = Interval(-0.01, 0.01)
U₀ = Interval(2.0, 2.0)
prob = @ivp(x' = f!(x), dim: 5, x(0) ∈ X₀ × W₀ × U₀);

# solve it
##sol = solve(prob, T=2.0);

# ## Specifications
#
# The verification problem here is that of target reachability. For an initial set
# of ``x₁ ∈ [0.35, 0.45]``, ``x₂ ∈ [0.45, 0.55]``, ``x₃ ∈ [0.25, 0, 35]``, it is
# required to prov that the system converges to ``x ∈ [−0.032, 0.032]^3`` within ``2s``.

# ## Results

# ## References

# [1] [Dong-Hae Yeom and Young Hoon Joo. Control lyapunov function design by
# cancelling input singularity. International Journal of Fuzzy Logic and
# Intelligent Systems, 12, 06 2012](http://www.koreascience.or.kr/article/JAKO201220962918909.page).
#
