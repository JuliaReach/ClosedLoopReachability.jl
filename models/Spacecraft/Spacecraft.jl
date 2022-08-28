# # 2D Spacecraft Docking
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Spacecraft.ipynb)

module Spacecraft  #jl

using ClosedLoopReachability
using ClosedLoopReachability: ProjectionPostprocessing

# ## Model

# model parameters
const m = 12.0
const n = 0.001027
const n² = n^2

@taylorize function Spacecraft!(du, u, p, t)
    x = u[1]
    y = u[2]
    x_dot = u[3]
    y_dot = u[4]
    Fx = u[5]
    Fy = u[6]

    ## differential equations for the quadrotor
    du[1] = x_dot
    du[2] = y_dot
    du[3] = 2 * n * y_dot + 3 * n² * x + Fx / m
    du[4] = -2 * n * x_dot + Fy / m
    du[5] = zero(Fx)
    du[6] = zero(Fy)
    return du
end

# ## Specification

# the ONNX network format is not supported by ONNX.jl
using MAT
controller = read_nnet_mat(@modelpath("Spacecraft", "model.mat"), act_key="act_fcns");
# using ONNX
# controller = read_nnet_onnx(  # MatMul not supported by ONNX.jl
#     ONNX.load(@modelpath("Spacecraft", "model.onnx"),
#     zeros(Float32, 4)));
# controller = read_nnet_onnx(  # Sub not supported by ONNX.jl
#     ONNX.load(@modelpath("Spacecraft", "bias_model.onnx"),
#     zeros(Float32, 4)));

## The initial states according to the specification are:
X₀ = Hyperrectangle([88, 88, 0.0, 0],
##                     [18, 18, 0.28, 0.28]  # original set
                    [1, 1, 0.01, 0.01]  # reduced set
                   )
U₀ = ZeroSet(2);

## The system has 4 state variables and 2 control variables:
vars_idx = Dict(:states=>1:4, :controls=>5:6)
ivp = @ivp(x' = Spacecraft!(x), dim: 6, x(0) ∈ X₀ × U₀)
period = 1.0;  # control period
postprocessing = ProjectionPostprocessing(1:2)

prob = ControlledPlant(ivp, controller, vars_idx, period;
                       postprocessing=postprocessing);

## Safety specification
T = 40.0  # time horizon

function predicate(v::AbstractVector)
    x, y, x_dot, y_dot = v
    return sqrt(x_dot^2 + y_dot^2) <= 0.2 + 2 * n * sqrt(x^2 + y^2)
end;

## sufficient check with interval arithmetic

function predicate_suff(R)
    x, y, x_dot, y_dot = convert(IntervalBox, box_approximation(R))
    return sqrt(x_dot^2 + y_dot^2) <= 0.2 + 2 * n * sqrt(x^2 + y^2)
end;

## sufficient check with zonotopes

# import Polyhedra
# function predicate_suff2(R)
#     Z = overapproximate(R, Zonotope)
#     Z = project(Z, 1:4)  # project to state variables
#     for v in vertices_list(Z)
#         if !predicate(v)
#             return false
#         end
#     end
#     return true
# end

predicate_sol = sol -> all(predicate_suff(R) for F in sol for R in F);

# ## Results

# To integrate the ODE, we use the Taylor-model-based algorithm:
alg = TMJets(abstol=1e-10, orderT=5, orderQ=1, adaptive=false);

# To propagate sets through the neural network, we use the `DeepZ` algorithm:
alg_nn = DeepZ()

function benchmark(; silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed solve(prob, T=T, alg_nn=alg_nn, alg=alg)
    sol = res_sol.value
    silent || print_timed(res_sol)

    ## Next we check the property:
    silent || println("property checking")
    res_pred = @timed predicate_sol(sol)
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
res = @timed simulate(prob, T=T, trajectories=1, include_vertices=true)
sim = res.value
print_timed(res);

## property check for simulation

# using ClosedLoopReachability: EnsembleSimulationSolution
# function predicate(sim_sols::EnsembleSimulationSolution)
#     for sim in sim_sols.solutions
#         for traj in sim.trajectory
#             for i in length(traj)
#                 v = traj[i]
#                 v2 = v[1:4]  # project to state variables
#                 !predicate(v2) && return false
#             end
#         end
#     end
#     return true
# end

# Finally we plot the results:

using Plots

function plot_helper(fig, vars; show_simulation::Bool=true)
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
end

vars = (1, 2)
fig = plot(xlab="x", ylab="y")
plot_helper(fig, vars)

savefig("Spacecraft-x-y.png")

#-

vars = (3, 4)
fig = plot(xlab="x'", ylab="y'")
plot_helper(fig, vars)

savefig("Spacecraft-x'-y'.png")

#-

end  #jl
nothing  #jl
