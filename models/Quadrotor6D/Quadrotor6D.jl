# # Quadrotor6D
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Quadrotor6D.ipynb)

module Quadrotor6D  #jl

using ClosedLoopReachability
import YAML

# ## Model

# model parameters
const g = 9.81  # gravity
period = 0.2  # control period
T = 30 * period  # time horizon

@taylorize function Quadrotor6D!(dx, x, p, t)
    if t <= 10 * period
        bx = 0.25
        by = -0.25
        bz = 0.0
    elseif t <= 20 * period
        bx = 0.25
        by = 0.25
        bz = 0.0
    elseif t <= 25 * period
        bx = 0.0
        by = -0.25
        bz = 0.0
    elseif t <= 30 * period
        bx = -0.25
        by = 0.25
        bz = 0.0
    end

    # px = x[1]
    # py = x[2]
    # pz = x[3]
    vx = x[4]
    vy = x[5]
    vz = x[6]
    θ  = x[7]
    ϕ  = x[8]
    τ  = x[9]

    dx[1] = vx - bx
    dx[2] = vy - by
    dx[3] = vz - bz
    dx[4] = one(vx) * (g * tan(θ))
    dx[5] = one(vy) * (-g * tan(ϕ))
    dx[6] = one(vz) * (τ - g)
    dx[7] = zero(ϕ)
    dx[8] = zero(θ)
    dx[9] = zero(τ)
    return dx
end

# ## Specification

# load the network
network = read_YAML(@modelpath("Quadrotor6D", "tanh20x20.yml"));

# build custom controller that returns one of eight input signals
function controller_conversion(x)
    # normalized network input
    _f1 = 0.2 * x[1]
    _f2 = 0.2 * x[2]
    _f3 = 0.2 * x[3]
    _f4 = 0.1 * x[4]
    _f5 = 0.1 * x[5]
    _f6 = 0.1 * x[6]
    x_normalized = [_f1, _f2, _f3, _f4, _f5, _f6]

    # propagate normalized input through network
    u = network(x_normalized)
    _f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8 = u

    # define some parameters
    _f9 = _f1 - _f2
    _f10 = _f1 - _f3
    _f11 = _f1 - _f4
    _f12 = _f1 - _f5
    _f13 = _f1 - _f6
    _f14 = _f1 - _f7
    _f15 = _f1 - _f8
    _f16 = _f2 - _f3
    _f17 = _f2 - _f4
    _f18 = _f2 - _f5
    _f19 = _f2 - _f6
    _f20 = _f2 - _f7
    d28 = _f2 - _f8
    d34 = _f3 - _f4
    d35 = _f3 - _f5
    d36 = _f3 - _f6
    d37 = _f3 - _f7
    d38 = _f3 - _f8
    d45 = _f4 - _f5
    d46 = _f4 - _f6
    d47 = _f4 - _f7
    d48 = _f4 - _f8
    d56 = _f5 - _f6
    d57 = _f5 - _f7
    d58 = _f5 - _f8
    d67 = _f6 - _f7
    d68 = _f6 - _f8
    d78 = _f7 - _f8

    # the network output gets categorized into one of seven classes
    if _f15 <= 0 && d28 <= 0 && d38 <= 0 && d48 <= 0 && d58 <= 0 && d68 <= 0 && d78 <= 0
        u1 = 0.1
        u2 = 0.1
        u3 = 11.81
    elseif _f14 <= 0 && _f20 <= 0 && d37 <= 0 && d47 <= 0 && d57 <= 0 && d67 <= 0 && d78 >= 0
        u1 = 0.1
        u2 = 0.1
        u3 = 7.81
    elseif _f13 <= 0 && _f19 <= 0 && d36 <= 0 && d46 <= 0 && d56 <= 0 && d67 >= 0 && d68 >= 0
        u1 = 0.1
        u2 = -0.1
        u3 = 11.81
    elseif _f12 <= 0 && _f18 <= 0 && d35 <= 0 && d45 <= 0 && d56 >= 0 && d57 >= 0 && d58 >= 0
        u1 = 0.1
        u2 = -0.1
        u3 = 7.81
    elseif _f11 <= 0 && _f17 <= 0 && d34 <= 0 && d45 >= 0 && d46 >= 0 && d47 >= 0 && d48 >= 0
        u1 = -0.1
        u2 = 0.1
        u3 = 11.81
    elseif _f10 <= 0 && _f16 <= 0 && d34 >= 0 && d35 >= 0 && d36 >= 0 && d37 >= 0 && d38 >= 0
        u1 = -0.1
        u2 = 0.1
        u3 = 7.81
    elseif _f9 <= 0 && _f16 >= 0 && _f17 >= 0 && _f18 >= 0 && _f19 >= 0 && _f20 >= 0 && d28 >= 0
        u1 = -0.1
        u2 = -0.1
        u3 = 11.81
    elseif _f9 >= 0 && _f10 >= 0 && _f11 >= 0 && _f12 >= 0 && _f13 >= 0 && _f14 >= 0 && _f15 >= 0
        u1 = -0.1
        u2 = -0.1
        u3 = 7.81
    else
        error("invalid network output $u")
    end
    return [u1, u2, u3]
end
controller = BlackBoxController(controller_conversion)

## The initial states according to the specification are:
X₀ = Hyperrectangle(zeros(6), [0.05, 0.05, 0, 0, 0, 0])
U₀ = Hyperrectangle(low=[-0.1, -0.1, 7.81], high=[0.1, 0.1, 11.81]);

## The system has 12 state variables and 3 control variables:
vars_idx = Dict(:states=>1:6, :controls=>7:9)
ivp = @ivp(x' = Quadrotor6D!(x), dim: 9, x(0) ∈ X₀ × U₀)

prob = ControlledPlant(ivp, controller, vars_idx, period);

# # ## Results

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

# # We also compute some simulations:

import DifferentialEquations

println("simulation")
res = @timed simulate(prob, T=T, trajectories=10, include_vertices=false)
sim = res.value
print_timed(res);

# # Finally we plot the results:

using Plots
import DisplayAs

function plot_helper(fig, vars; show_simulation::Bool=true)
    # plot!(fig, sol, vars=vars, color=:white, lab="")  # to set the plot limits
    # plot!(fig, sol, vars=vars, color=:yellow, lab="")
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars = (1, 2)
fig = plot(xlab="x", ylab="y")
plot_helper(fig, vars)
# savefig("Quadrotor6D-x-y.png")

end  #jl
nothing  #jl
