# # Unicycle Car Model
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/models/Sherlock-Benchmark-10-Unicycle.ipynb)
#
# This benchmark is that of a unicycle model of a car [^1] taken from Benchmark
# 10 of the Sherlock tool. It models the dynamics of a car involving 4
# variables, specifically the ``x`` and ``y`` coordinates on a 2 dimensional
# plane, as well as velocity magnitude (speed) and steering angle.

# ## Model
#
# The model is a four-dimensional system given by the following equations:
#
# ```math
# \left\{ \begin{array}{lcl}
# \dot{x}_1 &=& x_4 \cos(x_3) \\
# \dot{x}_2 &=& x_4 \sin(x_3) \\
# \dot{x}_3 &=& u_2 \\
# \dot{x}_4 &=& u_1 + w
# \end{array} \right.
# ```
# where ``w`` is a bounded error in the range ``[−1e−4, 1e−4]``. A neural
# network controller was trained for this system. The trained network has 1
# hidden layer with 500 neurons. Note that the output of the neural network
# ``f(x)`` needs to be normalized in order to obtain ``(u_1, u_2)``, namely
# ``u_i = f(x)_i − 20``. The sampling time for this controller is 0.2s.

using NeuralNetworkAnalysis
using NeuralNetworkAnalysis: UniformAdditiveNormalization

# We model the error ``w`` as a nondeterministically assigned constant.
@taylorize function unicycle!(dx, x, p, t)
    x₁, x₂, x₃, x₄, w, u₁, u₂ = x

    dx[1] = x₄ * cos(x₃)
    dx[2] = x₄ * sin(x₃)
    dx[3] = u₂
    dx[4] = u₁ + w
    dx[5] = zero(x[5])
    dx[6] = zero(x[6])
    dx[7] = zero(x[7])
    return dx
end;

controller = read_nnet_mat(@modelpath("Sherlock-Benchmark-10-Unicycle", "controllerB_nnv.mat");
                           act_key="act_fcns");

# ## Specification
#
# The verification problem here is that of reachability. For an initial set of,
# ``x_1 ∈ [9.5,9.55], x_2 ∈ [−4.5,−4.45], x_3 ∈ [2.1,2.11], x_4 ∈ [1.5,1.51]``,
# its is required to prove that the system reaches the set
# ``x_1 ∈ [−0.6,0.6], x_2 ∈ [−0.2,0.2], x_3 ∈ [−0.06,0.06], x_4 ∈ [−0.3,0.3]``
# within a time window of 10s.

X₀ = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5, -1e-4], high=[9.55, -4.45, 2.11, 1.51, 1e-1]);
U₀ = ZeroSet(2);
vars_idx = Dict(:state_vars=>1:4, :input_vars=>[5], :control_vars=>6:7);
ivp = @ivp(x' = unicycle!(x), dim: 7, x(0) ∈ X₀ × U₀);

period = 0.2;  # control period
T = 10.0;  # time horizon

control_normalization = UniformAdditiveNormalization(-20.0);

prob = ControlledPlant(ivp, controller, vars_idx, period;
                       normalization=control_normalization);

target_set = Hyperrectangle(zeros(4), [0.6, 0.2, 0.06, 0.3]);
# TODO spec: [x[1], x[2], x[3], x[4]] ∈ target_set for some 0 ≤ t ≤ T

# ## Results

alg = TMJets(abstol=1e-12, orderT=12, orderQ=2);
alg_nn = Ai2();

# @time sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg);  # TODO uncomment once the analysis works

# We also compute some simulations:
import DifferentialEquations
@time sim = simulate(prob, T=T; trajectories=10, include_vertices=true);

# Finally we plot the results
using Plots
import DisplayAs
vars = (0, 1);
fig = plot();
# plot!(fig, sol, vars=vars, lab="");  # TODO uncomment once the analysis works
xlims!(0, T)
ylims!(0, 10)
plot_simulation!(fig, sim; vars=vars, color=:red, lab="");
fig = DisplayAs.Text(DisplayAs.PNG(fig))

# ## References

# [^1] Souradeep Dutta, Xin Chen, and Sriram Sankaranarayanan. *Reachability
# analysis for neural feedback systems using regressive polynomial rule
# inference.* In [Proceedings of the 22nd ACMInternational Conference on Hybrid
# Systems: Computation and Control, HSCC 2019, Montreal,QC, Canada,
# April 16-18, 2019., pages 157–168, 2019](https://dl.acm.org/doi/abs/10.1145/3302504.3311807).
