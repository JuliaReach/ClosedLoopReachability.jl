# # Airplane
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Airplane.ipynb)
#
# This example consists of a dynamical system that is a simple model of a flying
# airplane. There are 12 state variables:
#
# ```math
# \begin{aligned}
# [x, y, z, u, v, w, ϕ, θ, ψ, r, p, q]
# \end{aligned}
# ```
#
# where ``(x, y, z)`` is the position of the C.G., ``(u, v, w)`` are the
# components of velocity in ``(x, y, z)`` directions, ``(p, q, r)`` are body
# rotation rates, and ``(ϕ, θ, ψ)`` are the Euler angles.
#
# ## Model
#
# The equations of motion are reduced to:
#
# ```math
# \left\{ \begin{array}{lcl}
# u̇ &=& − g sin(\theta) + \frac{F_x}{m} - qw + rv \\
# v̇ &=& g cos(\theta) sin(\phi) + \frac{F_y}{m} - ru + pw \\
# ẇ &=& g cos(\theta) cos(\phi) + \frac{F_z}{m} - pv + qu \\
# I_x \dot p + I_{xz} \dot r &=& M_x - (I_z - I_y) qr - I_{xz} pq \\
# I_y \dot q &=& M_y - I_{xz}(r^2 - p^2) - (I_x - I_z) pr \\
# I_{xz} \dot p + I_z \dot r &=& M_z - (I_y - I_x) qp - I_{xz} rq
# \end{array} \right.
# ```
#
# The mass of the airplane is denoted with m and ``I_x`` , ``I_y`` , ``I_z``
# and ``I_{xz}`` are the moment of inertia with respect to the indicated axis.
# The control parameters include three force components ``F_x`` , ``F_y`` and
# ``F_z`` and three moment components ``M_x`` , ``M_y`` , ``M_z`` . Notice that
# for simplicity we have assumed the aerodynamic forces are absorbed in the
# ``F``’s. In addition to these six equations, we have six additional kinematic
# equations:
#
# ```math
# \begin{bmatrix}
# \dot x \\ \dot y \\ \dot z
# \end{bmatrix}
# =
# \begin{bmatrix}
# cos(\psi) & -sin(\psi) & 0 \\
# sin(\psi) & cos(\psi) & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# \begin{bmatrix}
# cos(\theta) & 0 & sin(\theta) \\
# 0 & 1 & 0 \\
# -sin(\theta) & 0 & cos(\theta)
# \end{bmatrix}
# \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & cos(\phi) & -sin(\phi) \\
# 0 & sin(\phi) & cos(\phi)
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
# 1 & tan(\theta) sin(\phi) & tan(\theta) cos(\phi) \\
# 0 & cos(\phi) & -sin(\phi) \\
# 0 & sec(\theta) sin(\phi) & sec(\theta) cos(\phi)
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

Tψ = ψ -> [cos(ψ)  -sin(ψ)  0;
           sin(ψ)   cos(ψ)  0;
                0        0  1];

Tθ = θ -> [ cos(θ)  0  sin(θ);
                 0  1       0;
           -sin(θ)  0  cos(θ)];

Tϕ = ϕ -> [1       0        0;
           0  cos(ϕ)  -sin(ϕ);
           0  sin(ϕ)   cos(ϕ)];

Rϕθ = (ϕ, θ) -> [1  tan(θ) * sin(ϕ)  tan(θ) * cos(ϕ);
                 0           cos(θ)          -sin(ϕ);
                 0  sec(θ) * sin(ϕ)  sec(θ) * cos(ϕ)];

# alternative matrix with only sin/cos but with postprocessing
# Rϕθ_ = (ϕ, θ) -> [cos(θ)  sin(θ) * sin(ϕ)   sin(θ) * cos(ϕ);
#                        0  cos(θ) * cos(ϕ)  -cos(θ) * sin(ϕ);
#                        0           sin(ϕ)            cos(ϕ)];

# model constants
const m = 1.0
const g = 1.0
const Ix = 1.0  # not used (terms are simplified)
const Iy = 1.0  # not used (terms are simplified)
const Iz = 1.0  # not used (terms are simplified)
const Ixz = 0.0  # not used (terms are simplified)

@taylorize function airplane!(dx, x, p, t)
    _x, y, z, u, v, w, ϕ, θ, ψ, r, _p, q, Fx, Fy, Fz, Mx, My, Mz = x

    T_ψ = Tψ(ψ)
    T_θ = Tθ(θ)
    T_ϕ = Tϕ(ϕ)
    mat_1 = T_ψ * T_θ * T_ϕ
    xyz = mat_1 * vcat(u, v, w)

    mat_2 = Rϕθ(ϕ, θ)
    ## mat_2 = 1 / cos(θ) * Rϕθ_(ϕ, θ)  # alternative matrix with postprocessing
    ϕθψ = mat_2 * vcat(_p, q, r)

    dx[1] = xyz[1]
    dx[2] = xyz[2]
    dx[3] = xyz[3]
    dx[4] = -g * sin(θ) + Fx / m - q * w + r * v
    dx[5] = g * cos(θ) * sin(ϕ) + Fy / m - r * u + _p * w
    dx[6] = g * cos(θ) * cos(ϕ) + Fz / m - _p * v + q * u
    dx[7] = ϕθψ[1]
    dx[8] = ϕθψ[2]
    dx[9] = ϕθψ[3]
    dx[10] = Mx  # simplified term
    dx[11] = My  # simplified term
    dx[12] = Mz  # simplified term
    dx[13] = zero(Fx)
    dx[14] = zero(Fy)
    dx[15] = zero(Fz)
    dx[16] = zero(Mx)
    dx[17] = zero(My)
    dx[18] = zero(Mz)
end;

controller = read_nnet(@modelpath("Airplane", "controller_airplane.nnet"));

# ## Specifications
#
# ``x_2`` should be in ``[−0.5, 0.5]`` and ``x_7, x_8, x_9`` should be in ``[-1.0,1.0]``.

X₀ = 0.2 * Hyperrectangle(low=[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          high=[0.0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]);
U₀ = ZeroSet(6);

vars_idx = Dict(:state_vars=>1:12, :control_vars=>13:18);
ivp = @ivp(x' = airplane!(x), dim: 18, x(0) ∈ X₀ × U₀);

period = 0.1;  # control period
T = 20 * period;  # time horizon

prob = ControlledPlant(ivp, controller, vars_idx, period);

# TODO spec: x[2] ∈ [−0.5, 0.5] and [x[7], x[8], x[9]] ∈ [-1, 1]^3 for all t

# ## Results

alg = TMJets(abs_tol=1e-15, orderT=7, orderQ=1);
alg_nn = Ai2();

# @time sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg);

# We also compute some simulations:
using DifferentialEquations
@time sim = simulate(prob, T=T; trajectories=10, include_vertices=true);

# Finally we plot the results
using Plots
import DisplayAs
vars = (0, 1);
fig = plot();
# plot!(fig, sol, vars=vars, lab="");  # TODO uncomment once the analysis works
xlims!(0, T)
ylims!(0, 10)
plot_simulation!(fig, sim; vars=vars, color=:red, lab="");
fig = DisplayAs.Text(DisplayAs.PNG(fig))

#= TODO old code related to property checking
solz = overapproximate(sol, Zonotope);
@show ρ(sparsevec([2], [1.0], 18), solz)
@show -ρ(sparsevec([2], [-1.0], 18), solz)
[radius(solz.F.ext[:controls][i][1]) for i in 1:3]
radius(overapproximate(solz.F.ext[:controls][3][1], Hyperrectangle))
=#
