# # Non-linear Cart-Pole
#
# The model is a version of the cart-pole introduced by Barto, Sutton, and Anderson in [1].

# ## Model
#
# The dynamics of the cart-pole system are described as follows:
#
# ```math
# \begin{aligned}
# \ddot{x} &= \dfrac{u + mlω^2sin(θ)}{mt} - \dfrac{ml(g sin(θ)- cos(θ))
#             (\dfrac{u + mlω^2sin(θ)}{mt})}{l(\dfrac{4}{3} - \dfrac{m cos(θ)^2}{mt})}
#             \dfrac{cos(θ)}{mt}    \\
# \ddot{θ} &= \dfrac{(g sin(θ)- cos(θ))(\dfrac{u + mlω^2sin(θ)}{mt})}{l(\dfrac{4}{3}
#             - \dfrac{m cos(θ)^2}{mt})} \dfrac{cos(θ)}{mt}
# \end{aligned}
# ```
# where ``u ∈ {−10,10}`` is the input force, which either pushes the cart left
# or right, ``g = 9.8`` is gravity, ``m = 0.1`` is the pole’s mass, ``l = 0.5``
# is half the pole’s length , ``mt = 1.1`` is the total mass, ``x`` is the
# position of the cart, θ is the angle of the pendulum with respect to the
# positive y-axis, ``v = \dot{x}`` is the linear velocity of the cart, and
# ``ω = ̇θ`` is the angular velocity of the pendulum. The controller has four
# inputs ``(x, \dot{x}, θ, ̇θ)``, four layers with ``[24,48,12,2]`` neurons
# respectively, and two outputs. The two outputs are then compared, and the
# input sent to the plant depends on which output index has the greatest value.
#  Thus, as an example if ``output_1 > output_2`` then the input force supplied
# to the plant is 10.  However if ``output_1 < output_2`` then the input supplied
# to the plant is -10.

using NeuralNetworkAnalysis

@taylorize function cartpole!(du, u, p, t)
    local f, m, l, mt, g = 10, 0.1, 0.5, 1.1, 9.8
    sinθ = sin(u[3])
    cosθ = cos(u[3])
    aux = (f + m*l*u[4]^2*sinθ) / mt
    aux2 = l*(4/3 - m*cosθ^2/mt)
    aux3 = (g*sinθ- cosθ) * aux / aux2
    aux4 = cosθ/mt
    aux5 = m*l*aux3

    du[1] = u[2]
    du[2] = aux - aux5 * aux4
    du[3] = u[4]
    du[4] = aux3 * aux4

    return du
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[-0.05, -0.05, -0.05, -0.05], high=[0.05, 0.05, 0.05, 0.05])

prob = @ivp(x' = cartpole!(x), dim: 4, x(0) ∈ X₀)

# solve it
## sol = solve(prob, T=1.0, alg=TMJets(max_steps=20_000, abs_tol=1e-10));

# ## Specifications
#
# For this benchmark, the verification objective is to demonstrate that the
# pole will eventually reach the upward position and that it will remain there.
#  In other words, the goal is to achieve a value of ``θ = 0`` and stay there.
#  Some other specifications to be met are, for at least 12 seconds,
# ``x ∈ [-2.4,2.4]`` and ``θ ∈ [-15,15]`` degrees. The initial conditions for
# all state variables were chosen uniformly at random between ``[-0.05, 0.05]``.

# ## Results

# ## References

# [1] [A. G. Barto, R. S. Sutton, and C. W. Anderson. Neuronlike adaptive
# elements that can solve difficult learning control problems.
# IEEE Transactions on Systems, Man, and Cybernetics, SMC-13(5):834–846,
# Sep. 1983](https://ieeexplore.ieee.org/abstract/document/6313077).
#
