# # Quadrotor
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Quadrotor.ipynb)

module Quadrotor  #jl

using ClosedLoopReachability
using ClosedLoopReachability: SingleEntryVector

# ## Model

# model parameters
const g = 9.81
const m = 1.4
const Jx = 0.054
const Jy = 0.054
const Jz = 0.104
const Cyzx = (Jy - Jz) / Jx
const Czxy = (Jz - Jx) / Jy
const Cxyz = (Jx - Jy) / Jz
const τψ = 0.0
const Tz = τψ / Jz

@taylorize function Quadrotor!(dx, x, p, t)
    x₁ = x[1]
    x₂ = x[2]
    x₃ = x[3]
    x₄ = x[4]
    x₅ = x[5]
    x₆ = x[6]
    x₇ = x[7]
    x₈ = x[8]
    x₉ = x[9]
    x₁₀ = x[10]
    x₁₁ = x[11]
    x₁₂ = x[12]
    u₁ = x[13]
    u₂ = x[14]
    u₃ = x[15]

    F₁ = g + u₁ / m
    Tx = u₂ / Jx
    Ty = u₃ / Jy

    ## some abbreviations
    sx7 = sin(x₇)
    cx7 = cos(x₇)
    sx8 = sin(x₈)
    cx8 = cos(x₈)
    sx9 = sin(x₉)
    cx9 = cos(x₉)

    sx7sx9 = sx7 * sx9
    sx7cx9 = sx7 * cx9
    cx7sx9 = cx7 * sx9
    cx7cx9 = cx7 * cx9
    sx7cx8 = sx7 * cx8
    cx7cx8 = cx7 * cx8
    sx7_cx8 = sx7 / cx8
    cx7_cx8 = cx7 / cx8

    x4cx8 = cx8 * x₄

    p11 = sx7_cx8 * x₁₁
    p12 = cx7_cx8 * x₁₂
    xdot9 = p11 + p12

    ## differential equations for the quadrotor
    dx[1] = (cx9 * x4cx8 + (sx7cx9 * sx8 - cx7sx9) * x₅) + (cx7cx9 * sx8 + sx7sx9) * x₆
    dx[2] = (sx9 * x4cx8 + (sx7sx9 * sx8 + cx7cx9) * x₅) + (cx7sx9 * sx8 - sx7cx9) * x₆
    dx[3] = (sx8 * x₄ - sx7cx8 * x₅) - cx7cx8 * x₆
    dx[4] = (x₁₂ * x₅ - x₁₁ * x₆) - g * sx8
    dx[5] = (x₁₀ * x₆ - x₁₂ * x₄) + g * sx7cx8
    dx[6] = (x₁₁ * x₄ - x₁₀ * x₅) + (g * cx7cx8 - F₁)
    dx[7] = x₁₀ + sx8 * xdot9
    dx[8] = cx7 * x₁₁ - sx7 * x₁₂
    dx[9] = xdot9
    dx[10] = Cyzx * (x₁₁ * x₁₂) + Tx
    dx[11] = Czxy * (x₁₀ * x₁₂) + Ty
    dx[12] = Cxyz * (x₁₀ * x₁₁) + Tz

    dx[13] = zero(u₁)
    dx[14] = zero(u₂)
    dx[15] = zero(u₃)
    return dx
end

@taylorize function Quadrotor_simplified!(dx, x, p, t)
    x₁ = x[1]
    x₂ = x[2]
    x₃ = x[3]
    x₄ = x[4]
    x₅ = x[5]
    x₆ = x[6]
    x₇ = x[7]
    x₈ = x[8]
    x₉ = x[9]
    x₁₀ = x[10]
    x₁₁ = x[11]
    u₁ = x[13]
    u₂ = x[14]
    u₃ = x[15]

    F₁ = g + u₁ / m
    Tx = u₂ / Jx
    Ty = u₃ / Jy

    ## some abbreviations
    sx7 = sin(x₇)
    cx7 = cos(x₇)
    sx8 = sin(x₈)
    cx8 = cos(x₈)
    sx9 = sin(x₉)
    cx9 = cos(x₉)

    sx7sx9 = sx7 * sx9
    sx7cx9 = sx7 * cx9
    cx7sx9 = cx7 * sx9
    cx7cx9 = cx7 * cx9
    sx7cx8 = sx7 * cx8
    cx7cx8 = cx7 * cx8
    sx7_cx8 = sx7 / cx8
    cx7_cx8 = cx7 / cx8

    x4cx8 = cx8 * x₄

    xdot9 = sx7_cx8 * x₁₁

    ## differential equations for the quadrotor
    dx[1] = (cx9 * x4cx8 + (sx7cx9 * sx8 - cx7sx9) * x₅) + (cx7cx9 * sx8 + sx7sx9) * x₆
    dx[2] = (sx9 * x4cx8 + (sx7sx9 * sx8 + cx7cx9) * x₅) + (cx7sx9 * sx8 - sx7cx9) * x₆
    dx[3] = (sx8 * x₄ - sx7cx8 * x₅) - cx7cx8 * x₆
    dx[4] = - x₁₁ * x₆ - g * sx8
    dx[5] = x₁₀ * x₆ + g * sx7cx8
    dx[6] = (x₁₁ * x₄ - x₁₀ * x₅) + (g * cx7cx8 - F₁)
    dx[7] = x₁₀ + sx8 * xdot9
    dx[8] = cx7 * x₁₁
    dx[9] = xdot9
    dx[10] = Tx
    dx[11] = Ty
    dx[12] = zero(x[12])

    dx[13] = zero(u₁)
    dx[14] = zero(u₂)
    dx[15] = zero(u₃)
    return dx
end

# ## Specification

# the simpler network format is more efficient to parse than the ONNX format
controller = read_nnet_polar(@modelpath("Quadrotor", "quad_controller_3_64"));
# using ONNX
# controller_onnx = read_nnet_onnx(
#     ONNX.load(@modelpath("Quadrotor", "quad_controller_3_64_torch.onnx"),
#     zeros(Float32, 12)));
# @assert controller_onnx == controller

## The initial states according to the specification are:
X₀ = Hyperrectangle(zeros(12),
##                     [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0]  # original set
                    [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0, 0, 0, 0, 0, 0]  # reduced set
                   )
U₀ = ZeroSet(3);

## The system has 12 state variables and 3 control variables:
vars_idx = Dict(:states=>1:12, :controls=>13:15)
ivp = @ivp(x' = Quadrotor_simplified!(x), dim: 15, x(0) ∈ X₀ × U₀)
period = 0.1;  # control period

prob = ControlledPlant(ivp, controller, vars_idx, period);

## Safety specification
T = 5.0  # time horizon

target_states = HPolyhedron([HalfSpace(SingleEntryVector(3, 15, -1.0), -0.94),
                             HalfSpace(SingleEntryVector(3, 15, 1.0), 1.06)])
predicate = X -> X ⊆ target_states
predicate_sol = sol -> any(predicate(R) for F in sol for R in F);

## sufficient check: only look at the final time point
predicate_R_tend = R -> overapproximate(R, Zonotope, tend(R)) ⊆ target_states
predicate_R_all = R -> R ⊆ target_states
predicate_sol_suff = sol -> predicate_R_all(sol[end]);

# ## Results

# To integrate the ODE, we use the Taylor-model-based algorithm:
alg = TMJets(abstol=1e-8, orderT=5, orderQ=1, adaptive=false);

# To propagate sets through the neural network, we use the `DeepZ` algorithm:
alg_nn = DeepZ()


function benchmark(; silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed solve(prob, T=T, alg_nn=alg_nn, alg=alg)
    sol = res_sol.value
    silent || print_timed(res_sol)

    ## Next we check the property for an overapproximated flowpipe:
    silent || println("property checking")
    res_pred = @timed predicate_sol_suff(sol)
    silent || print_timed(res_pred)
    if res_pred.value
        silent || println("The property is satisfied.")
    else
        silent || println("The property may be violated.")
    end
    return sol
end

benchmark(silent=true)  # warm-up
res = @timed benchmark()  # benchmark
sol = res.value
println("total analysis time")
print_timed(res);

# We also compute some simulations:

import DifferentialEquations

println("simulation")
res = @timed simulate(prob, T=T, trajectories=10, include_vertices=false)
sim = res.value
print_timed(res);

# Finally we plot the results:

using Plots
import DisplayAs

function plot_helper(fig, vars; show_simulation::Bool=true)
    plot!(fig, sol, vars=vars, color=:white, lab="")  # to set the plot limits
    plot!(fig, project(target_states, vars), color=:green, lab="")
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars = (0, 3)
fig = plot(xlab="t", ylab="x₃")
plot_helper(fig, vars)
# savefig("Quadrotor-t-x3.png")

end  #jl
nothing  #jl
