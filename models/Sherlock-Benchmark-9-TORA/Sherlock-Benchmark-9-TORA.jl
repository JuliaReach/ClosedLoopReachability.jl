# # Translational Oscillations by a Rotational Actuator (TORA)
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Sherlock-Benchmark-9-TORA.ipynb)
#
# This model consists of a cart attached to a wall with a spring, and is free
# to move a friction-less surface. The car itself has a weight attached to an arm
# inside it, which is free to rotate about an axis as described in the figure below.
# This serves as the control input, in order to stabilize the cart at ``x = 0``.
#
# ## Model
#
# The model is four dimensional, given by the following equations:
#
# ```math
# \left\{ \begin{array}{lcl}
#       \dot{x}_1 &=& x_2 \\
#       \dot{x}_2 &=& -x_1 + 0.1 \sin x_3 \\
#       \dot{x}_3 &=& x_4  \\
#       \dot{x}_4 &=& u
# \end{array} \right.
# ```
#
# A neural network controller was trained for this system, using data-driven model predictive
# controller proposed in [^DJST18]. The trained network had 3 hidden layers, with 100 neurons in each
# layer making a total of 300 neurons. Note that the output of the neural network $f(x)$ needs to
# be normalized in order to obtain $u$, namely $u = f(x) - 10$. The sampling time for this controller
# was 1s.

using NeuralNetworkAnalysis

@taylorize function benchmark9!(dx, x, p, t)
    x₁, x₂, x₃, x₄, u = x

    aux = 0.1 * sin(x₃)
    dx[1] = x₂
    dx[2] = -x₁ + aux
    dx[3] = x₄
    dx[4] = u
    dx[5] = zero(u)
end

# ## Requirements

# The verification problem here is that of safety.
# For an initial set of $x_1 ∈ [0.6, 0.7]$, $x_2 ∈ [−0.7, −0.6]$, $x_3 ∈ [−0.4, −0.3]$,
# and $x_4 ∈ [0.5, 0.6]$, the system states stay within the box $x ∈ [−2, 2]^4$ for a time window of 20s.

# ## Initial-value problem

X₀ = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
U = ZeroSet(1)
ivp = @ivp(x' = benchmark9!(x), dim: 5, x(0) ∈ X₀×U);

vars_idx = Dict(:state_vars=>1:4, :input_vars=>[], :control_vars=>[5]);

using MAT
path = joinpath(@modelpath("Sherlock-Benchmark-9-TORA", "controllerTora.mat"))
controller = read_nnet_mat(path, act_key="act_fcns");

using NeuralNetworkAnalysis: UniformAdditiveNormalization

τ = 1.0 # control period
N = UniformAdditiveNormalization(-10.0)
plant = ControlledPlant(ivp, controller, vars_idx, τ, N);

safe_states = BallInf(zeros(2), 2.0);

# ## Simulations

using DifferentialEquations, Plots

simulations, controls, inputs = simulate(plant, T=20.0, trajectories=20);

#-

fig = plot(xlab="x₁", ylab="x₂")
plot!(fig, safe_states, color=:white, linecolor=:black, lw=5.0)
for simulation in simulations
    plot!(fig, simulation, vars=(1, 2))
end
plot!(fig, project(X₀, 1:2), lab="X₀")

#-

fig = plot(xlab="x₂", ylab="x₄")
plot!(fig, safe_states, color=:white, linecolor=:black, lw=5.0)
for simulation in simulations
    plot!(fig, simulation, vars=(2, 4))
end
plot!(fig, project(X₀, [2, 4]), lab="X₀")

#-

fig = plot(xlab="x₂", ylab="x₃")
plot!(fig, safe_states, color=:white, linecolor=:black, lw=5.0)
for simulation in simulations
    plot!(fig, simulation, vars=(2, 3))
end
plot!(fig, project(X₀, [2, 3]), lab="X₀")

#-

# Here we plot the control functions for each run:

tdom = range(0, 20, length=length(first(controls)))
fig = plot(xlab="time", ylab="u")
[plot!(fig, tdom, [c[1] for c in controls[i]], lab="") for i in eachindex(controls)]
fig

# ## Flowpipe computation

# ## References

# [^DJST18]: Souradeep Dutta, Susmit Jha, Sriram Sankaranarayanan, and Ashish Tiwari.
#            *Learning and verification of feedback control systems using feedforward neural networks.*
#            IFAC-PapersOnLine, 51(16):151 – 156, 2018. 6th IFAC Conference on Analysis and Design of Hybrid Systems ADHS2018.
