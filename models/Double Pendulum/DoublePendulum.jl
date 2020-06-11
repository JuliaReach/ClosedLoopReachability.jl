# # Double Pendulum
#
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`DoublePendulum.ipynb`](__NBVIEWER_ROOT_URL__/../DoublePendulum.ipynb)
#
#
# This example includes a double-link pendulum with equal point masses ``m``
# at the end of connected mass-less links of length ``L``. Both links are
# actuated with torques ``T_1`` and ``T_2`` and we assume viscous friction
# exists with a coefficient of ``c``. [1].

# ## Model
#
# For this casestudy, the ego car is set to travel at a set speed ``V_{set} = 30``
# and maintains a safe distance ``D_{safe}`` from the lead car.  The car’s
# dynamics are described as follows:
#
# ```math
# \begin{aligned}
# 2\ddot \theta_1 + \ddot \theta_2 cos(\theta_2 - \theta_1) - \ddot \theta^2_2 sin(\theta_2 - \theta_1) - 2 \frac{g}{L}sin\theta_1 + \frac{c}{mL^2}\dot\theta_1 &= \frac{1}{mL^2}T_1 \\
# \ddot \theta_1 cos(\theta_2 - \theta_1) + \ddot \theta_2 + \ddot \theta^2_1 sin(\theta_2 - \theta_1) - \frac{g}{L}sin\theta_2 + \frac{c}{mL^2}\dot\theta_2 &= \frac{1}{mL^2}T_2
# \end{aligned}
# ```
# where ``\theta_1`` and ``\theta_2`` are the angles that links make with the
# upward vertical axis. The state is:
# ```math
# \begin{aligned}
# [\theta_1, \theta_2, \dot \theta_1, \dot \theta_2]
# \end{aligned}
# ```
# The angular velocity and acceleration of links are donoted with
# ``\dot \theta_1``, ``\dot \theta_2``, ``\ddot \theta_1`` and ``\ddot \theta_2``
# and ``g``` is the gravitational acceleration.


using NeuralNetworkAnalysis

@taylorize function DoublePendulum!(dx, x, p, t)
    x₁, x₂, x₃ = x
    dx[1] = x₁
    dx[2] = x₂
    dx[3] = x₃
end

# define the initial-value problem
##X₀ = Hyperrectangle(low=[...], high=[...])

##prob = @ivp(x' = DoublePendulum!(x), dim: ?, x(0) ∈ X₀)

# solve it
##sol = solve(prob, T=0.1);

# ## Specifications
#

# ## Results

# ## References

# [1] Amir Maleki, Chelsea Sidrane, May 16, 2020, [Benchmark Examples for
# AINNCS-2020](https://github.com/amaleki2/benchmark_closedloop_verification/blob/master/AINNC_benchmark.pdf).
#
