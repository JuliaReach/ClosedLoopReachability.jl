# # Adaptive Cruise Controller (ACC)
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/ACC.ipynb)
#
# The Adaptive Cruise Control (ACC) benchmark is a system that tracks a set
# velocity and maintains a safe distance from a lead vehicle by adjusting
# the longitudinal acceleration of an ego vehicle.
# The neural network computes optimal control actions while satisfying safe
# distance, velocity, and acceleration constraints using model predictive
# control (MPC) [^QB00].

# ## Model
#
# For this case study, the ego car is set to travel at a set speed $V_{set} = 30$
# and maintains a safe distance $D_{safe}$ from the lead car.  The car’s dynamics
# are described as follows:
#
# ```math
# \left\{ \begin{array}{lcl}
# \dot{x}_{lead}(t) &=& v_{lead}(t) \\
# \dot{v}_{lead}(t) &=& γ_{lead}(t) \\
# \dot{γ}_{lead}(t) &=& -2γ_{lead}(t) + 2a_{lead}(t) - μv_{lead}^2(t)  \\
# \dot{x}_{ego}(t) &=& v_{ego}(t) \\
# \dot{v}_{ego}(t) &=& γ_{ego}(t) \\
# \dot{γ}_{ego}(t) &=& -2γ_{ego}(t) + 2a_{ego}(t) - μv_{ego}^2(t)
# \end{array} \right.
# ```
# where ``x_i`` is the position, ``v_i`` is the velocity, ``γ_i`` is the
# acceleration of the car, ``a_i`` is the acceleration control input applied
# to the car, and ``μ = 0.0001`` is the friction parameter, where
# ``i ∈ \{ego, lead\}``. For this benchmark we are given four neural network
# controllers with 3, 5, 7, and 10 hidden layers of 20 neurons each, but only evaluate
# the controller with 5 hidden layers. All of hem have the same number of inputs
# ``(V_{set}, T_{gap}, v_{ego}, D_{rel}, v_{rel})`` and one output (``a_{ego}``).

using NeuralNetworkAnalysis

const μ = 0.0001 # friction parameter
const a_lead = -2.0 # acceleration control input applied to the lead vehicle

@taylorize function ACC!(dx, x, p, t)
    x_lead = x[1] # lead car position
    v_lead = x[2] # lead car velocity
    γ_lead = x[3] # lead car internal state

    x_ego = x[4] # ego car position
    v_ego = x[5] # ego car velocity
    γ_ego = x[6] # ego car internal state
    a_ego = x[7] # ego car acceleration control input

    ## lead car dynamics
    dx[1] = v_lead
    dx[2] = γ_lead
    dx[3] = 2 * (a_lead - γ_lead) - μ * v_lead^2

    ## ego car dynamics
    dx[4] = v_ego
    dx[5] = γ_ego
    dx[6] = 2 * (a_ego - γ_ego) - μ * v_ego^2
    dx[7] = zero(a_ego)
    return dx
end

# ## Specifications
#
# The verification objective of this system is that given a scenario where both
# cars are driving safely, the lead car suddenly slows down with``a_{lead} = -2``.
# We want to check whether there is a collision in the following 5 seconds.
# A control period of 0.1 seconds is used.
#
# Formally, the safety specification can be expressed as:
# ```math
#     D_{rel} = x_{lead} - x_{ego} ≥ D_{safe},
# ```
# where ``D_{safe} = D_{default} + T_{gap} × v_{ego}``,
# ``T_{gap} = 1.4`` sec and ``D_{default} = 10``.

# The uncertain initial conditions are chosen to be:
#
# - ``x_{lead}(0) ∈ [90,110], v_{lead}(0) ∈ [32,32.2], γ_{lead}(0) = γ_{ego}(0) = 0``
# - ``v_{ego}(0) ∈ [30, 30.2], x_{ego} ∈ [10,11]``

# ## Results

# The initial states according to the specification are:
X₀ = Hyperrectangle(low=[90, 32, 0, 10, 30, 0], high=[110, 32.2, 0, 11, 30.2, 0])
U₀ = Singleton([-1.0]); # gets overwritten

# The system has 6 state variables and 1 control variable:
vars_idx = Dict(:state_vars=>1:6, :control_vars=>7)
ivp = @ivp(x' = ACC!(x), dim: 7, x(0) ∈ X₀×U₀);

# We will evaluate the controller which has 5 hidden layers.
using MAT
path = joinpath(@modelpath("ACC", "controller_5_20.mat"))
controller = read_nnet_mat(path, act_key="act_fcns")

period = 0.1
prob = ControlledPlant(ivp, controller, vars_idx, period);

# To integrate the ODE we use the Taylor model based algorithm:
alg = TMJets(abs_tol=1e-12, orderT=8, orderQ=2);

# To propagate sets over the neural network we use the `Ai2` algorithm:
alg_nn = Ai2();

# We now solve the controlled system:
#sol = solve(plant, T=0.2, alg_nn=solver, alg=alg);

# ## References

# [^QB00]: S. Joe Qin and Thomas A. Badgwell. *An overview of nonlinear model predictive control applications.* In [Nonlinear  Model  Predictive  Control, pages 369–392, Basel, 2000. Birkh ̈auser Basel](https://link.springer.com/chapter/10.1007/978-3-0348-8407-5_21).
