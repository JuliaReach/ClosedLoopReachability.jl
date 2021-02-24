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

@taylorize function airplane!(dx, x, p, t)
    _x, y, z, u, v, w, ϕ, θ, ψ, r, _p, q, Fx, Fy, Fz, Mx, My, Mz = x

    T_ψ = [cos(ψ)   -sin(ψ)   0.;
           sin(ψ)   cos(ψ)    0.;
           0.       0.        1.]

    T_θ = [cos(θ)   0.    sin(θ);
           0.       1.    0.;
           -sin(θ)  0.    cos(θ)]

    T_ϕ = [1.    0.      0.   ;
           0.  cos(ϕ)  -sin(ϕ);
           0.  sin(ϕ)   cos(ϕ)]

    mat_1 = T_ψ * T_θ * T_ϕ

    mat_2 = [cos(θ)  sin(θ) * sin(ϕ)       sin(θ) * cos(ϕ);
             0.          cos(θ) * cos(ϕ)  -cos(θ) * sin(ϕ);
             0.          sin(ϕ)               cos(ϕ)     ]
    mat_2 = 1 / cos(θ) * mat_2

    a1 = [u; v; w]
    a2 = mat_1 * a1

    _dx = a2[1]
    dy = a2[2]
    dz = a2[3]

    a3 = [_p; q; r]
    a4 = mat_2 * a3

    dϕ = a4[1]
    dθ = a4[2]
    dψ = a4[3]

    du = -sin(θ)          + Fx - q * w + r * v
    dv =  cos(θ) * sin(ϕ) + Fy - r * u + _p * w
    dw =  cos(θ) * cos(ϕ) + Fz - _p * v + q * u

    dp = Mx
    dq = My
    dr = Mz

    dx[1] = _dx
    dx[2] = dy
    dx[3] = dz
    dx[4] = du
    dx[5] = dv
    dx[6] = dw
    dx[7] = dϕ
    dx[8] = dθ
    dx[9] = dψ
    dx[10] = dp
    dx[11] = dq
    dx[12] = dr
    dx[13] = zero(Fx)
    dx[14] = zero(Fy)
    dx[15] = zero(Fz)
    dx[16] = zero(Mx)
    dx[17] = zero(My)
    dx[18] = zero(Mz)
end

# ## Specifications
#
# ``x_2`` should be in ``[−0.5, 0.5]`` and ``x_7, x_8, x_9`` should be in ``[-1.0,1.0]``.

# ## Results

X₀ = Hyperrectangle(low=[0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], high=[0., 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]);
X₀ = 0.2 * X₀
X₀ = overapproximate(X₀, Hyperrectangle)

U₀ = rand(Hyperrectangle, dim=6) # ignored
prob = @ivp(x' = airplane!(x), dim: 18, x(0) ∈ X₀ × U₀);
vars_idx = Dict(:state_vars=>1:12, :input_vars=>[], :control_vars=>13:18);
period = 0.1

#=
plant = ControlledPlant(prob, controller, vars_idx, period);
alg = TMJets(abs_tol=1e-15, orderT=7, orderQ=1)
solver = Ai2z()
@time sol = solve(plant, T=2.0, alg_nn=solver, alg=alg)
solz = overapproximate(sol, Zonotope);


@show ρ(sparsevec([2], [1.0], 18), solz)
@show -ρ(sparsevec([2], [-1.0], 18), solz)

[radius(solz.F.ext[:controls][i][1]) for i in 1:3]


radius(overapproximate(solz.F.ext[:controls][3][1], Hyperrectangle))

=#

# ## References

# [1] Amir Maleki, Chelsea Sidrane, May 16, 2020, [Benchmark Examples for
# AINNCS-2020](https://github.com/amaleki2/benchmark_closedloop_verification/blob/master/AINNC_benchmark.pdf).
#
