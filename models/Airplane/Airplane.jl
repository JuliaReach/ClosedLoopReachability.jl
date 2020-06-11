# # Airplane
#
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`Airplane.ipynb`](__NBVIEWER_ROOT_URL__/../Airplane.ipynb)
#
#
# This example consists of a dynamical system that is a simple model of
# a flying airplane. The state is:
#
# ```math
# \begin{aligned}
# [x, y, z, u, v, w, φ, θ, ψ, r, p, q]
# \end{aligned}
# ```
#
# where (x, y, z) is the position of the C.G., (u, v, w) are the components of
# velocity in (x, y, z) directions, (p, q, r) are body rotation rates, and (φ, θ, ψ)
# are the Euler angles.
# ## Model
#
# The equations of motion are reduced to:
#
# ```math
# \begin{aligned}
# u̇ &= − g sin \theta + \frac{F_x}{m} - qw + rv \\
# v̇ &= g cos \theta sin \phi + \frac{F_y}{m} - ru + pw \\
# ẇ &= g cos \theta cos \phi + \frac{F_z}{m} - pv + qu \\
# I_x \dot p + I_{xz} \dot r &= M_x - (I_z - I_y) qr - I_{xz} pq \\
# I_y \dot q &= M_y - I_{xz}(r^2 - p^2) - (I_x - I_z) pr \\
# I_{xz} \dot p + I_z \dot r &= M_z - (I_y - I_x) qp - I_{xz} rq
# \end{aligned}
# ```
# The mass of the airplane is denoted with m and ``I_x`` , ``I_y`` , ``I_z`` and ``I_{xz}`` are the
# moment of inertia with respect to the indicated axis. The controls
# parameters include three force components ``F_x`` , ``F_y`` and ``F_z`` and three moment
# components ``M_x`` , ``M_y`` , ``M_z`` . Notice that for simplicity we have assumed the
# aerodynamic forces are absorbed in the ``F``’s. In addition to these six equations,
# we have six additional kinematic equations:
#
# ```math
# \begin{bmatrix}
# \dot x \\ \dot y \\ \dot z
# \end{bmatrix}
# =
# \begin{bmatrix}
# cos\psi & -sin\psi & 0 \\
# sin\psi & cos\psi & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# \begin{bmatrix}
# cos\theta & 0 & sin\theta \\
# 0 & 1 & 0 \\
# -sin\theta & 0 & cos\theta
# \end{bmatrix}
# \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & cos\phi & -sin\phi \\
# 0 & sin\phi & cos\phi
# \end{bmatrix}
# ```
#
# and
#
# ```math
# \begin{bmatrix}
# \phi \\ \theta \\ \psi
# \end{bmatrix}
# =
# \begin{bmatrix}
# 1 & tan\theta sin\phi & tan\theta cos\phi \\
# 0 & cos\phi & -sin\phi \\
# 0 & sec\theta sin\phi & sec\theta cos\phi
# \end{bmatrix}
# \begin{bmatrix}
# p \\ q \\ r
# \end{bmatrix}
# ```
#
# For the simplicity of control design, the parameters have been chosen to
# have some nominal dimensionless values:
# ``m = 1``, ``I_x = I_y = I_z = 1``, ``I_{xz} = 0`` and ``g = 1``.

using NeuralNetworkAnalysis

@taylorize function Airplane!(dx, x, p, t)
    x₁, x₂, x₃ = x
    dx[1] = x₁
    dx[2] = x₂
    dx[3] = x₃
end

# define the initial-value problem
##X₀ = Hyperrectangle(low=[...], high=[...])

##prob = @ivp(x' = Airplane!(x), dim: ?, x(0) ∈ X₀)

# solve it
##sol = solve(prob, T=0.1);

# ## Specifications
#

# ## Results

# ## References

# [1] Amir Maleki, Chelsea Sidrane, May 16, 2020, [Benchmark Examples for
# AINNCS-2020](https://github.com/amaleki2/benchmark_closedloop_verification/blob/master/AINNC_benchmark.pdf).
#
