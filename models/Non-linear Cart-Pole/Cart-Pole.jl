# # Non-linear Cart-Pole
#
# The model is a version of the cart-pole introduced by Barto, Sutton, and Anderson in [xxx].

# ## Model
#
# The dynamics of the cart-pole system are described as follows:
#
# ```math
# \begin{aligned}
# \ddot{x} &= \dfrac{u + mlω^2*sin(θ)}{mt} - \dfrac{ml(g sin(θ)- cos(θ))
#             (\dfrac{u + mlω^2*sin(θ)}{mt})}{l(\dfrac{4}{3} - \dfrac{m cos(θ)^2}{mt})}
#             \dfrac{cos(θ)}{mt}    \\
# \ddot{θ} &= \dfrac{g sin(θ)- cos(θ)(\dfrac{u + mlω^2*sin(θ)}{mt})}{l(\dfrac{4}{3}
#             - \dfrac{m cos(θ)^2}{mt})} \dfrac{cos(θ)}{mt}
# \end{aligned}
# ```
# where ``u ∈ {−10,10}`` is the input force, which either pushes the cart left
# or right, ``g = 9.8`` is gravity, ``m = 0.1`` is the pole’s mass, ``l = 0.5``
# is half the pole’s length , ``mt = 1.1`` is the total mass, ``x`` is the
# position of the cart, θ is the angle of the pendulum with respect to the
# positive y-axis, ``v = ̇x`` is the linear velocity of the cart, and
# ``ω = ̇θ`` is the angular velocity of the pendulum. The controller has four
# inputs ``(x, ̇x, θ, ̇θ)``, four layers with ``[24,48,12,2]`` neurons
# respectively, and two outputs. The two outputs are then compared, and the
# input sent to the plant depends onwhich output index has the greatest value.
#  Thus, as an example if ``output1 > output2`` then the input force supplied
# to the plant is 10.  However if ``output1 < output2`` then the input supplied
# to the plant is -10.

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
@time sol = solve(prob, T=1.0, alg=TMJets(max_steps=20_000, abs_tol=1e-10));

# ## Specifications
#
# The verification problem here is that of reachability. For an initial set of,
# ``x1 ∈ [9.5,9.55], x2 ∈ [−4.5,−4.45], x3 ∈ [2.1,2.11], x4 ∈ [1.5,1.51]``, it
# is required to prove that the system reaches the set ``x1 ∈ [−0.6,0.6], x2 ∈``
# [−0.2,0.2], x3 ∈ [−0.06,0.06], x4 ∈ [−0.3,0.3]`` within a time window of 10s.

# ## Results

# ## References

#
#
