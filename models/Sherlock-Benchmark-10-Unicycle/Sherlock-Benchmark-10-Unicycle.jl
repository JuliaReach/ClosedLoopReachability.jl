# # Unicycle Car Model
#
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`Sherlock-Benchmark-10.ipynb`](__NBVIEWER_ROOT_URL__/../Sherlock-Benchmark-10-Unicycle.ipynb)
#
#
# This benchmark is that of a unicycle model of a car [^1]. It models the
# dynamics of a car involving 4 variables, specifically the ``x`` and ``y``
# coordinates on a 2 dimensional plane, as well as velocity magnitude (speed)
# and steering angle.

# ## Model
#
# The model is a four-dimensional system given by the following equations:
#
# ```math
# \begin{aligned}
# \dot{x}_1 &= x_4 cos(x_3) \\
# \dot{x}_2 &= x_4 sin(x_3) \\
# \dot{x}_3 &= u_2 \\
# \dot{x}_4 &= u_1 + w
# \end{aligned}
# ```
# where ``w`` is a bounded error in the range ``[−1e−4, 1e−4]``. A neural network
# controller was trained for this system, using a model predictive controller
# as a "demonstrator" or "teacher". The trained network has 1 hidden layer,
# with 500 neurons. The sampling time for this controller was 0.2s.

using NeuralNetworkAnalysis

@taylorize function benchmark10!(dx, x, p, t)
    u1, u2 = one(x[5]), one(x[5])
    dx[1] = x[4] * cos(x[3])
    dx[2] = x[4] * sin(x[3])
    dx[3] = u2
    dx[4] = u1 + x[5]
    dx[5] = zero(x[5]) # w
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5, -1e-4], high=[9.55, -4.45, 2.11, 1.51, 1e-1])

prob = @ivp(x' = benchmark10!(x), dim: 5, x(0) ∈ X₀)

# TODO add normalization of control inputs to u = NN(x) .- 20

# solve it
##sol = solve(prob, T=10.0);

# ## Specifications
#
# The verification problem here is that of reachability. For an initial set of,
# ``x_1 ∈ [9.5,9.55], x_2 ∈ [−4.5,−4.45], x_3 ∈ [2.1,2.11], x_4 ∈ [1.5,1.51]``,
# its is required to prove that the system reaches the set
# ``x_1 ∈ [−0.6,0.6], x_2 ∈ [−0.2,0.2], x_3 ∈ [−0.06,0.06], x_4 ∈ [−0.3,0.3]``
# within a time window of 10s.

# ## Results

# ## References

# [^1] Souradeep Dutta, Xin Chen, and Sriram Sankaranarayanan. *Reachability
# analysis for neural feedback systems using regressive polynomial rule
# inference.* In [Proceedings of the 22nd ACMInternational Conference on Hybrid
# Systems: Computation and Control, HSCC 2019, Montreal,QC, Canada,
# April 16-18, 2019., pages 157–168, 2019](https://dl.acm.org/doi/abs/10.1145/3302504.3311807).
#
