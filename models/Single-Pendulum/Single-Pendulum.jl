# # Single Inverted Pendulum
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`singlePendulum.ipynb`](__NBVIEWER_ROOT_URL__/../Single-Pendulum.ipynb)
#
# This is a classical inverted pendulum. A ball of mass ``m`` is
# attached to a massless beam of length ``L``.  The beam is actuated with a
# torque ``T`` and we assume viscous friction exists with a coefficient of
# ``c``.

# ## Model
#
# The governing equation of motion can be obtained as:
#
# ```math
# \ddot\theta = \dfrac{g}{L} \sin\theta + \dfrac{1}{m L^2} (T - c\dot\theta)
# ```
# where ``θ`` is the angle that link makes with the upward vertical axis.
# The state vector is ``[θ, θ']``.
# Controllers are trained using behavior cloning. Here, a neural network is
# trained to replicate expert demonstrations [^1].

using NeuralNetworkAnalysis

#models_dir = normpath(@__DIR__, "..", "..", "..", "models")
#path = joinpath(models_dir, "SinglePendulum", )
controller = read_nnet(@relpath "controller_single_pendulum.nnet")

# model constants
m = 0.5
L = 0.5
c = 0.
g = 1.0

gL = g/L
mL = 1/m/L^2

function single_pendulum!(dx, x, params, t)
    dx[1] = x[2]
    dx[2] = gL * sin(x[1]) + mL*(x[3] - c*x[2])
    dx[3] = zero(x[1]) # T = x[3]
    return dx
end

# define the initial-value problem
X0 = Hyperrectangle([1.1, 0.1], [0.1, 0.1]);
U0 = Universe(1)

ivp = @ivp(x' = single_pendulum!(x), dim: 3, x(0) ∈ X0 × U0);
vars_idx = Dict(:state_vars=>1:2, :input_vars=>[], :control_vars=>3);
period = 0.05

prob = ControlledPlant(ivp, controller, vars_idx, period);
alg = TMJets(abs_tol=1e-12, orderT=5, orderQ=2)
alg_nn = Ai2()

# solve it
#@time sol = solve(prob, T=1.0, alg_nn=alg_nn, alg=alg)
#solz = overapproximate(sol, Zonotope);

# ## Specifications
#
# The (continuous-time) safety specification is: ``10 ≤ t ≤ 20``, ``θ ∈ [0,1]``.

# ## Results

#alg = TMJets(abs_tol=1e-10, orderT=8, orderQ=2)
#solver = Ai2()

# solve it
#sol = solve(plant, T=0.1, Tsample=0.05, alg_nn=solver, alg=alg);

# ## References

# [^1]: Johnson, T. T., Manzanas Lopez, D., Musau, P., Tran, H. D., Botoeva, E.,
#       Leofante, F., ... & Huang, C. (2020). ARCH-COMP20 Category Report: Artificial
#       Intelligence and Neural Network Control Systems (AINNCS) for Continuous and
#       Hybrid Systems Plants. EPiC Series in Computing, 74.
#       (https://easychair.org/publications/open/Jvwg)
