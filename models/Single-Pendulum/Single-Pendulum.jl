# # Single Inverted Pendulum
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Single-Pendulum.ipynb)
#
# This is a classical inverted pendulum. A ball of mass ``m`` is attached to a
# massless beam of length ``L``.  The beam is actuated with a torque ``T`` and
# we assume viscous friction exists with a coefficient of ``c``.

# ## Model
#
# The governing equation of motion can be obtained as:
#
# ```math
# \ddot\theta = \dfrac{g}{L} \sin(\theta) + \dfrac{1}{m L^2} (T - c\dot\theta)
# ```
# where ``θ`` is the angle that the link makes with the upward vertical axis.
# The state vector is ``[θ, θ']``.

using NeuralNetworkAnalysis

# model constants
const m = 0.5;
const L = 0.5;
const c = 0.0;
const g = 1.0;
const gL = g/L;
const mL = 1/(m*L^2);

function single_pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = gL * sin(x[1]) + mL*(x[3] - c*x[2])
    dx[3] = zero(x[3])
    return dx
end;

controller = read_nnet(@modelpath("Single-Pendulum",
                                  "controller_single_pendulum.nnet"));

# ## Specification
#
# The initial set is
# ```math
# [\theta, \dot\theta] = [1, 1.2] \times [0, 0.2].
# ```
# The safety specification is: ``10 ≤ t ≤ 20``, ``θ ∈ [0, 1]``.

X0 = Hyperrectangle([1.1, 0.1], [0.1, 0.1]);
U0 = ZeroSet(1);
ivp = @ivp(x' = single_pendulum!(x), dim: 3, x(0) ∈ X0 × U0);
vars_idx = Dict(:state_vars=>1:2, :control_vars=>3);

period = 0.05;  # control period
T = 20.0;  # time horizon

prob = ControlledPlant(ivp, controller, vars_idx, period);

# TODO spec: ``x[1] ∈ [0, 1]`` for all ``10 ≤ t ≤ 20``

# ## Results

alg = TMJets(abstol=1e-10, orderT=4, orderQ=2);
alg_nn = Ai2();
# # @time sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg);  # TODO uncomment once the analysis works

# We also compute some simulations:
import DifferentialEquations
@time sim = simulate(prob, T=T; trajectories=10, include_vertices=true);

# Finally we plot the results
using Plots
import DisplayAs
vars = (0, 1);
fig = plot();
# # plot!(fig, sol, vars=vars, lab="");  # TODO uncomment once the analysis works
xlims!(0, T);
ylims!(-0.1, 1.3);
plot_simulation!(fig, sim; vars=vars, color=:red, lab="");
fig = DisplayAs.Text(DisplayAs.PNG(fig))
