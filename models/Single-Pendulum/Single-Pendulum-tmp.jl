using NeuralNetworkAnalysis

controller = read_nnet(@modelpath("Single-Pendulum", "controller_single_pendulum.nnet"))

# model constants
const m = 0.5
const L = 0.5
const c = 0.
const g = 1.0
const gL = g/L
const mL = 1/(m*L^2)

function single_pendulum!(dx, x, params, t)
    dx[1] = x[2]
    dx[2] = gL * sin(x[1]) + mL*(x[3] - c*x[2])
    dx[3] = zero(x[1]) # T = x[3]
    return dx
end

# define the initial-value problem
X0 = Hyperrectangle([1.1, 0.1], [0.1, 0.1]);
U0 = ZeroSet(1)

ivp = @ivp(x' = single_pendulum!(x), dim: 3, x(0) ∈ X0 × U0);
vars_idx = Dict(:state_vars=>1:2, :control_vars=>3);
period = 0.05

prob = ControlledPlant(ivp, controller, vars_idx, period);
# alg = TMJets(abs_tol=1e-12, orderT=5, orderQ=2)
alg = TMJets(abs_tol=1e-10, orderT=8, orderQ=3)

T = 0.1

# solve it
alg_nn = VertexSolver(x -> overapproximate(x, Interval))
@time solV = NeuralNetworkAnalysis.solve(prob, T=T, alg_nn=alg_nn, alg=alg)
# solz = overapproximate(sol, Zonotope)

alg_nn = Ai2()
@time solA = NeuralNetworkAnalysis.solve(prob, T=T, alg_nn=alg_nn, alg=alg)

using Plots
plot(overapproximate(solV.F.ext[:XXs][1], Zonotope), vars=(1,2))
plot!(overapproximate(solA.F.ext[:XXs][1], Zonotope), vars=(1,2))
