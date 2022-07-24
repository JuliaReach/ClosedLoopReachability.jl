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
# where $(x, y, z)$ is the position of the center of gravity (C.G.),
# $(u, v, w)$ are the components of velocity in $(x, y, z)$ directions,
# $(p, q, r)$ are body rotation rates, and $(ϕ, θ, ψ)$ are the Euler angles.
#
# See also the file `airplane.png` for a visualization of the meaning of the
# coordinates with respect to the airplane.

module Airplane  #jl

using ClosedLoopReachability

# The following option determines whether the falsification settings should be
# used or not. The falsification settings are sufficient to show that the safety
# property is violated. Concretely, we start from an initial point and use a
# smaller time step.
const falsification = true;

# ## Model
#
# The equations of motion are reduced to:
#
# ```math
# \left\{ \begin{array}{lcl}
# u̇ &=& − g \sin(\theta) + \frac{F_x}{m} - qw + rv \\
# v̇ &=& g \cos(\theta) \sin(\phi) + \frac{F_y}{m} - ru + pw \\
# ẇ &=& g \cos(\theta) \cos(\phi) + \frac{F_z}{m} - pv + qu \\
# I_x \dot p + I_{xz} \dot r &=& M_x - (I_z - I_y) qr - I_{xz} pq \\
# I_y \dot q &=& M_y - I_{xz}(r^2 - p^2) - (I_x - I_z) pr \\
# I_{xz} \dot p + I_z \dot r &=& M_z - (I_y - I_x) qp - I_{xz} rq
# \end{array} \right.
# ```
#
# The mass of the airplane is denoted with m and $I_x$ , $I_y$ , $I_z$
# and $I_{xz}$ are the moment of inertia with respect to the indicated axis.
# The control parameters include three force components $F_x$ , $F_y$ and
# $F_z$ and three moment components $M_x$ , $M_y$ , $M_z$ . Notice that
# for simplicity we have assumed the aerodynamic forces are absorbed in the
# $F$'s. In addition to these six equations, we have six additional kinematic
# equations:
#
# ```math
# \begin{bmatrix}
# \dot x \\ \dot y \\ \dot z
# \end{bmatrix}
# =
# \begin{bmatrix}
# \cos(\psi) & -\sin(\psi) & 0 \\
# \sin(\psi) & \cos(\psi) & 0 \\
# 0 & 0 & 1
# \end{bmatrix}
# \begin{bmatrix}
# \cos(\theta) & 0 & \sin(\theta) \\
# 0 & 1 & 0 \\
# -\sin(\theta) & 0 & \cos(\theta)
# \end{bmatrix}
# \begin{bmatrix}
# 1 & 0 & 0 \\
# 0 & \cos(\phi) & -\sin(\phi) \\
# 0 & \sin(\phi) & \cos(\phi)
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
# 1 & \tan(\theta) \sin(\phi) & \tan(\theta) \cos(\phi) \\
# 0 & \cos(\phi) & -\sin(\phi) \\
# 0 & \sec(\theta) sin(\phi) & \sec(\theta) \cos(\phi)
# \end{bmatrix}
# \begin{bmatrix}
# p \\ q \\ r
# \end{bmatrix}
# ```
#
# For the simplicity of control design, the parameters have been chosen to
# have some nominal dimensionless values:
# $m = 1$, $I_x = I_y = I_z = 1$, $I_{xz} = 0$ and $g = 1$.

Tψ = ψ -> [cos(ψ)  -sin(ψ)  0;
           sin(ψ)   cos(ψ)  0;
                0        0  1]

Tθ = θ -> [ cos(θ)  0  sin(θ);
                 0  1       0;
           -sin(θ)  0  cos(θ)]

Tϕ = ϕ -> [1       0        0;
           0  cos(ϕ)  -sin(ϕ);
           0  sin(ϕ)   cos(ϕ)]

Rϕθ = (ϕ, θ) -> [1  tan(θ) * sin(ϕ)  tan(θ) * cos(ϕ);
                 0           cos(θ)          -sin(ϕ);
                 0  sec(θ) * sin(ϕ)  sec(θ) * cos(ϕ)]

## alternative matrix with only sin/cos but with postprocessing
## Rϕθ_ = (ϕ, θ) -> [cos(θ)  sin(θ) * sin(ϕ)   sin(θ) * cos(ϕ);
##                        0  cos(θ) * cos(ϕ)  -cos(θ) * sin(ϕ);
##                        0           sin(ϕ)            cos(ϕ)]

## model constants
const m = 1.0
const g = 1.0

## unused constants (terms are simplified instead)
const Ix = 1.0
const Iy = 1.0
const Iz = 1.0
const Ixz = 0.0

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
    dx[10] = Mz  # simplified term
    dx[11] = Mx  # simplified term
    dx[12] = My  # simplified term
    dx[13] = zero(Fx)
    dx[14] = zero(Fy)
    dx[15] = zero(Fz)
    dx[16] = zero(Mx)
    dx[17] = zero(My)
    dx[18] = zero(Mz)
    return dx
end;

# ## Specifications

controller = read_nnet(@modelpath("Airplane", "controller_airplane.nnet"))

X₀ = Hyperrectangle(low=[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    high=[0.0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
if falsification
    ## choose a single point in the initial states (here: the top-most one)
    X₀ = Singleton(high(X₀))
end
U₀ = ZeroSet(6)

vars_idx = Dict(:states=>1:12, :controls=>13:18)
ivp = @ivp(x' = airplane!(x), dim: 18, x(0) ∈ X₀ × U₀)

period = 0.1  # control period

prob = ControlledPlant(ivp, controller, vars_idx, period);

# Safety specification: $[x_2, x_7, x_8, x_9] ∈ ±[0.5, 1, 1, 1]$.
if falsification
    k = 4  # falsification can run for a shorter time horizon
else
    k = 20
end
T = k * period  # time horizon

safe_states = concretize(CartesianProductArray([
    Universe(1), Interval(-0.5, 0.5),  # x_2 ∈ ± 0.5
    Universe(4), BallInf(zeros(3), 1.0),  # x_7, x_8, x_9 ∈ ± 1
    Universe(9)]))

## property for guaranteed violation
predicate = X -> isdisjoint(overapproximate(X, Hyperrectangle), safe_states)
predicate_sol = sol -> any(predicate(R) for F in sol for R in F);

# ## Results

import DifferentialEquations

alg = TMJets(abstol=1e-10, orderT=7, orderQ=1)
alg_nn = DeepZ()

function benchmark(; silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed solve(prob, T=T, alg_nn=alg_nn, alg=alg)
    sol = res_sol.value
    silent || print_timed(res_sol)

    ## Next we check the property for an overapproximated flowpipe:
    silent || println("property checking")
    solz = overapproximate(sol, Zonotope)
    res_pred = @timed predicate_sol(solz)
    silent || print_timed(res_pred)
    if res_pred.value
        silent || println("The property is violated.")
    else
        silent || println("The property may be satisfied.")
    end

    ## We also compute some simulations:
    silent || println("simulation")
    trajectories = falsification ? 1 : 50
    res_sim = @timed simulate(prob, T=T, trajectories=trajectories,
                              include_vertices=!falsification)
    sim = res_sim.value
    silent || print_timed(res_sim)

    return solz, sim
end

benchmark(silent=true)  # warm-up
res = @timed benchmark()  # benchmark
sol, sim = res.value
println("total analysis time")
print_timed(res);

# Finally we plot the results:

using Plots
import DisplayAs

## set more precise tolerance for plotting small sets correctly
ClosedLoopReachability.LazySets.set_ztol(Float64, 1e-9)

function plot_helper(fig, vars)
    if vars[1] == 0
        safe_states_projected = project(safe_states, [vars[2]])
        time = Interval(0, T)
        safe_states_projected = cartesian_product(time, safe_states_projected)
    else
        safe_states_projected = project(safe_states, vars)
    end
    plot!(fig, safe_states_projected, color=:lightgreen, lab="safe states")
    if !falsification && 0 ∉ vars
        plot!(fig, project(initial_state(prob), vars), lab="X₀")
    end
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    lab_sim = falsification ? "simulation" : ""
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars = (2, 7)
fig = plot(xlab="y", ylab="ϕ", leg=:bottomleft)
fig = plot_helper(fig, vars)
if falsification
    xlims!(-0.01, 0.65)
    ylims!(0.85, 1.01)
else
    xlims!(-0.55, 0.55)
    ylims!(-1.05, 1.05)
end
## savefig("Airplane-x2-x7.png")
fig

#-

vars = (8, 9)
fig = plot(xlab="θ", ylab="ψ", leg=:bottom)
fig = plot_helper(fig, vars)
if falsification
    xlims!(0.9, 1.01)
    ylims!(0.85, 1.01)
else
    xlims!(-1.05, 1.05)
    ylims!(-1.05, 1.05)
end
## savefig("Airplane-x8-x9.png")
fig

#-

end  #jl
nothing  #jl
