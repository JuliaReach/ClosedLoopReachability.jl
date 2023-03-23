# # Attitude Control
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/AttitudeControl.ipynb)

module AttitudeControl  #jl

using ClosedLoopReachability

# ## Model

@taylorize function AttitudeControl!(dx, x, p, t)
    ω₁ = x[1]
    ω₂ = x[2]
    ω₃ = x[3]
    ψ₁ = x[4]
    ψ₂ = x[5]
    ψ₃ = x[6]
    u₀ = x[7]
    u₁ = x[8]
    u₂ = x[9]

    aux = ψ₁^2 + ψ₂^2 + ψ₃^2

    dx[1] = 0.25 * (u₀ + ω₂ * ω₃)
    dx[2] = 0.5 * (u₁ - 3 * ω₁ * ω₃)
    dx[3] = u₂ + 2 * ω₁ * ω₂
    dx[4] = 0.5 * (  ω₂ * (aux - ψ₃)
                   + ω₃ * (aux + ψ₂)
                   + ω₁ * (aux + 1))
    dx[5] = 0.5 * (  ω₁ * (aux + ψ₃)
                   + ω₃ * (aux - ψ₁)
                   + ω₂ * (aux + 1))
    dx[6] = 0.5 * (  ω₁ * (aux - ψ₂)
                   + ω₂ * (aux + ψ₁)
                   + ω₃ * (aux + 1))
    dx[7] = zero(u₀)
    dx[8] = zero(u₁)
    dx[9] = zero(u₂)
    return dx
end

# ## Specification

# the simpler network format is more efficient to parse than the ONNX format
controller = read_POLAR(@modelpath("AttitudeControl", "CLF_controller_layer_num_3"));
# using ONNX
# controller_onnx = read_nnet_onnx(
#     ONNX.load(@modelpath("AttitudeControl", "attitude_control_3_64_torch.onnx"),
#     zeros(Float32, 6)));
# @assert controller_onnx == controller

## The initial states according to the specification are:
X₀ = Hyperrectangle(low=[-0.45, -0.55, 0.65, -0.75, 0.85, -0.65],
                    high=[-0.44, -0.54, 0.66, -0.74, 0.86, -0.64])
U₀ = ZeroSet(3);

## The system has 6 state variables and 3 control variables:
vars_idx = Dict(:states=>1:6, :controls=>7:9)
ivp = @ivp(x' = AttitudeControl!(x), dim: 9, x(0) ∈ X₀ × U₀)
period = 0.1;  # control period

prob = ControlledPlant(ivp, controller, vars_idx, period);

## Safety specification
T = 3.0  # time horizon
unsafe_states = cartesian_product(
    Hyperrectangle(low=[-0.2, -0.5, 0,   -0.7, 0.7, -0.4],
                   high=[0,   -0.4, 0.2, -0.6, 0.8, -0.2]),
    Universe(3))
predicate = X -> isdisjoint(overapproximate(X, Hyperrectangle), unsafe_states);

# ## Results

# To integrate the ODE, we use the Taylor-model-based algorithm:
alg = TMJets(abstol=1e-6, orderT=6, orderQ=1);

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
    res_pred = @timed predicate(sol)
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
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    unsafe_states_projected = project(unsafe_states, vars)
    plot!(fig, unsafe_states_projected, color=:red, alpha=:0.2,
          lab="unsafe states", leg=:topleft)
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end

vars = (1, 2)
fig = plot(xlab="ω₁", ylab="ω₂")
plot_helper(fig, vars)
# savefig("AttitudeControl-x1-x2.png")

#-

end  #jl
nothing  #jl
