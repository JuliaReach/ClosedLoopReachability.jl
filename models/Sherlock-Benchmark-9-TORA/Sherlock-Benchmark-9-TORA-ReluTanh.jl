# # Translational Oscillations by a Rotational Actuator (TORA)
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Sherlock-Benchmark-9-TORA.ipynb)

module TORA_ReluTanh  #jl

using ClosedLoopReachability, MAT
using ClosedLoopReachability: LinearMapPostprocessing

# This model consists of a cart attached to a wall with a spring. The cart is
# free to move on a friction-less surface. The car has a weight attached to an
# arm, which is free to rotate about an axis. This serves as the control input
# to stabilize the cart at $x = 0$.
#
# ## Model
#
# The model is four dimensional, given by the following equations:
#
# ```math
# \left\{ \begin{array}{lcl}
#       \dot{x}_1 &=& x_2 \\
#       \dot{x}_2 &=& -x_1 + 0.1 \sin(x_3) \\
#       \dot{x}_3 &=& x_4  \\
#       \dot{x}_4 &=& u
# \end{array} \right.
# ```
#
# A neural network controller was trained for this system. The trained network
# has 3 hidden layers, with 100 neurons in each layer (i.e., a total of 300
# neurons). Note that the output of the neural network $f(x)$ needs to be
# normalized in order to obtain $u$, namely $u = f(x) - 10$. The sampling time
# for this controller is 1s.

@taylorize function TORA!(dx, x, p, t)
    x₁, x₂, x₃, x₄, u = x

    aux = 0.1 * sin(x₃)
    dx[1] = x₂
    dx[2] = -x₁ + aux
    dx[3] = x₄
    dx[4] = u
    dx[5] = zero(u)
    return dx
end

path = @modelpath("Sherlock-Benchmark-9-TORA", "controllerToraReluTanh.mat")
controller = read_MAT(path, act_key="act_fcns");

# ## Specification

X₀ = Hyperrectangle(low=[-0.77, -0.45, 0.51, -0.3], high=[-0.75, -0.43, 0.54, -0.28])
U = ZeroSet(1)

vars_idx = Dict(:states=>1:4, :controls=>5)
ivp = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀ × U)

period = 0.5  # control period
control_postprocessing = LinearMapPostprocessing(11.0)  # control postprocessing

prob = ControlledPlant(ivp, controller, vars_idx, period;
                       postprocessing=control_postprocessing)

## Safety specification
T = 5.0  # time horizon
T_warmup = 2 * period  # shorter time horizon for dry run

goal_states_x1x2 = Hyperrectangle(low=[-0.1, -0.9], high=[0.2, -0.6])
goal_states = cartesian_product(goal_states_x1x2, Universe(3))
predicate = sol -> project(sol[end][end], [1, 2]) ⊆ goal_states_x1x2;

# ## Results

alg = TMJets(abstol=1e-10, orderT=8, orderQ=3)
alg_nn = DeepZ()

function benchmark(; T=T, silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed solve(prob, T=T, alg_nn=alg_nn, alg=alg)
    sol = res_sol.value
    silent || print_timed(res_sol)

    ## Next we check the property for an overapproximated flowpipe:
    silent || println("property checking")
    solz = overapproximate(sol, Zonotope)
    res_pred = @timed predicate(solz)
    silent || print_timed(res_pred)

    if res_pred.value
        silent || println("The property is satisfied.")
    else
        silent || println("The property may be violated.")
    end
    return solz
end

benchmark(T=T_warmup, silent=true)  # warm-up
res = @timed benchmark(T=T)  # benchmark
sol = res.value
println("total analysis time")
print_timed(res);

# We also compute some simulations:

import DifferentialEquations

println("simulation")
res = @timed simulate(prob, T=T; trajectories=1, include_vertices=true)
sim = res.value
print_timed(res);

# Finally we plot the results
using Plots
import DisplayAs

function plot_helper(fig, vars)
    if vars[1] == 0
        goal_states_projected = project(goal_states, [vars[2]])
        time = Interval(0, T)
        goal_states_projected = cartesian_product(time, goal_states_projected)
    else
        goal_states_projected = project(goal_states, vars)
    end
    plot!(fig, goal_states_projected, color=:cyan, lab="goal states")
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    lens!(fig, [0.0, 0.25], [-0.85, -0.7], inset = (1, bbox(0.4, 0.4, 0.3, 0.3)), lc=:black)
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars = (1, 2)
fig = plot(xlab="x₁", ylab="x₂")
fig = plot_helper(fig, vars)
## savefig("TORA-relutanhx1-x2.png")
fig

#-

end  #jl
nothing  #jl
