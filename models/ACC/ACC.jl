# # Adaptive Cruise Controller
#
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`ACC.ipynb`](__NBVIEWER_ROOT_URL__/../ACC.ipynb)
#
#
# The Adaptive Cruise Control (ACC) System simulates a system that tracks a
# set velocity and maintains a safe distance from a lead vehicle by adjusting
# the longitudinal acceleration of an ego vehicle.
# The neural network computes optimal control actions while satisfying safe
# distance, velocity, and acceleration constraints using model predictive
# control (MPC) [xxx].

# ## Model
#
# For this casestudy, the ego car is set to travel at a set speed ``V_{set} = 30``
# and maintains a safe distance ``D_{safe}`` from the lead car.  The car’s
# dynamics are described as follows:
#
# ```math
# \begin{aligned}
# \dot{x}_{lead}(t) &= v_{lead}(t) \\
# \dot{v}_{lead}(t) &= γ_{lead}(t) \\
# \dot{γ}_{lead}(t) &= -2γ_{lead}(t) + 2a_{lead}(t) - uv_{lead}^2(t)  \\
# \dot{x}_{ego}(t) &= v_{lead}(t) \\
# \dot{v}_{ego}(t) &= γ_{ego}(t) \\
# \dot{γ}_{ego}(t) &= -2γ_{ego}(t) + 2a_{ego}(t) - uv_{ego}^2(t)
# \end{aligned}
# ```
# where ``x_i`` is the position, ``v_i`` is the velocity, ``γ_i`` is the
# acceleration of the car, ``a_i`` is the acceleration control input applied
# to the car, and ``u = 0.0001`` is the friction control where
# ``i ∈ {ego, lead}``. For this benchmark we have developed four neural network
# controllers with 3, 5, 7, and 10 hidden layers of 20 neurons each. All of
# them have the same number of inputs ``(v_{set},T_{gap},v_{ego},D_{rel},v_{rel})``,
# and one output (``a_ego``).

using NeuralNetworkAnalysis

@taylorize function ACC!(dx, x, p, t)
    local u, a_lead, a_ego = 0.0001, -2., -1.
    x_lead, v_lead, γ_lead, x_ego, v_ego, γ_ego = x

    dx[1] = v_lead
    dx[2] = γ_lead
    dx[3] = 2 * (a_lead - γ_lead) - u * v_lead^2
    dx[4] = v_ego
    dx[5] = γ_ego
    dx[6] = 2 * (a_ego - γ_ego) - u * v_ego^2
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[90., 32, 0, 10, 30, 0], high=[110, 32.2, 0, 11, 30.2, 0])

prob = @ivp(x' = ACC!(x), dim: 6, x(0) ∈ X₀)

# solve it
##sol = solve(prob, T=0.1);

# ## Specifications
#
# The verification objective of this system is that given a scenario where both
# cars are driving safely, the lead car suddenly slows down with a ``lead = -2``.
#  We want to checkwhether there is a collision  in  the  following  5 seconds.
# Formally, this safety specification ofthe system can be expressed as
# ``D_{rel} = x_{lead} - x_{ego} ≥ D_{safe}``, where
# ``D_{safe} = D_{default} + T_{gap} × v_{ego}``, and
# ``T_{gap} = 1.4`` seconds and ``D_{default} = 10``. The initial conditions are:
# ``x_{lead}(0) ∈ [90,110], v_{lead}(0) ∈ [32,32.2], γ_{lead}(0) = γ_{ego}(0) = 0``,
# ``v_{ego}(0) ∈ [30, 30.2], x_{ego} ∈ [10,11]``.

# ## Results

# ## References

#
#
