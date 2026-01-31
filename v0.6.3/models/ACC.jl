module ACC

using ClosedLoopReachability
import OrdinaryDiffEq, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: FunctionPreprocessing
using Plots: plot, plot!

vars_idx = Dict(:states => 1:6, :controls => 7)

const u = 0.0001
const a_lead = -2.0

@taylorize function ACC!(dx, x, p, t)
    v_lead = x[2]  # lead car velocity
    γ_lead = x[3]  # lead car acceleration
    v_ego = x[5]  # ego car velocity
    γ_ego = x[6]  # ego car acceleration
    a_ego = x[7]  # ego car acceleration control input

    # Lead-car dynamics:
    dx[1] = v_lead
    dx[2] = γ_lead
    dx[3] = 2 * (a_lead - γ_lead) - u * v_lead^2

    # Ego-car dynamics:
    dx[4] = v_ego
    dx[5] = γ_ego
    dx[6] = 2 * (a_ego - γ_ego) - u * v_ego^2

    dx[7] = zero(a_ego)
    return dx
end;

path = @current_path("ACC", "ACC_controller_relu.polar")
controller_relu = read_POLAR(path)

path = @current_path("ACC", "ACC_controller_tanh.polar")
controller_tanh = read_POLAR(path);

v_set = 30.0
T_gap = 1.4
M = zeros(3, 6)
M[1, 5] = 1.0
M[2, 1] = 1.0
M[2, 4] = -1.0
M[3, 2] = 1.0
M[3, 5] = -1.0
function preprocess(X::LazySet)  # version for set computations
    Y1 = Singleton([v_set, T_gap])
    Y2 = linear_map(M, X)
    return cartesian_product(Y1, Y2)
end
function preprocess(X::AbstractVector)  # version for simulations
    Y1 = [v_set, T_gap]
    Y2 = M * X
    return vcat(Y1, Y2)
end
control_preprocessing = FunctionPreprocessing(preprocess);

period = 0.1;

X₀ = Hyperrectangle(low=[90, 32, 0, 10, 30, 0],
                    high=[110, 32.2, 0, 11, 30.2, 0])
U₀ = ZeroSet(1);

ivp = @ivp(x' = ACC!(x), dim: 7, x(0) ∈ X₀ × U₀)
problem(controller) = ControlledPlant(ivp, controller, vars_idx, period;
                                      preprocessing=control_preprocessing);

D_default = 10.0
d_rel = [1.0, 0, 0, -1, 0, 0, 0]
d_safe = [0, 0, 0, 0, T_gap, 0, 0]
d_prop = d_rel - d_safe
safe_states = HalfSpace(-d_prop, -D_default)

predicate(sol) = overapproximate(sol, Hyperrectangle) ⊆ safe_states

T = 5.0
T_warmup = 2 * period;  # shorter time horizon for warm-up run

algorithm_plant = TMJets(abstol=1e-3, orderT=5, orderQ=1);

algorithm_controller = DeepZ();

function benchmark(prob; T=T, silent::Bool=false)
    # Solve the controlled system:
    silent || println("Flowpipe construction:")
    res = @timed solve(prob; T=T, algorithm_controller=algorithm_controller,
                       algorithm_plant=algorithm_plant)
    sol = res.value
    silent || print_timed(res)

    # Check the property:
    silent || println("Property checking:")
    res = @timed predicate(sol)
    silent || print_timed(res)
    if res.value
        silent || println("  The property is satisfied.")
        result = "verified"
    else
        silent || println("  The property may be violated.")
        result = "not verified"
    end

    return sol, result
end;

function run(; use_relu_controller::Bool)
    if use_relu_controller
        println("# Running analysis with ReLU controller")
        prob = problem(controller_relu)
    else
        println("# Running analysis with tanh controller")
        prob = problem(controller_tanh)
    end

    # Run the verification benchmark:
    benchmark(prob; T=T_warmup, silent=true)  # warm-up
    res = @timed benchmark(prob; T=T)  # benchmark
    sol, result = res.value
    @assert (result == "verified") "verification failed"
    println("Total analysis time:")
    print_timed(res)

    # Compute some simulations:
    println("Simulation:")
    res = @timed simulate(prob; T=T, trajectories=10, include_vertices=true)
    sim = res.value
    print_timed(res)

    return sol, sim
end;

sol_relu, sim_relu = run(use_relu_controller=true);

sol_tanh, sim_tanh = run(use_relu_controller=false);

function plot_helper(sol, sim)
    fig = plot(leg=(0.4, 0.3))
    for F in sol, R in F
        # Subdivide the reach sets in time to obtain more precise plots:
        R = overapproximate(R, Zonotope; ntdiv=5)
        R_rel = linear_map(Matrix(d_rel'), R)
        plot!(fig, R_rel; vars=(0, 1), c=:red, lw=0, alpha=0.4)
    end

    solz = overapproximate(flowpipe(sol), Zonotope)
    fp_safe = affine_map(Matrix(d_safe'), solz, [D_default])
    plot!(fig, fp_safe; vars=(0, 1), c=:blue, lw=0, alpha=0.4)

    output_map_rel = x -> dot(d_rel, x)
    plot_simulation!(fig, sim; output_map=output_map_rel, color=:red, lab="Drel")

    output_map_safe = x -> dot(d_safe, x) + D_default
    plot_simulation!(fig, sim; output_map=output_map_safe, color=:blue, lab="Dsafe")

    plot!(fig; xlab="time")
    return fig
end;

fig = plot_helper(sol_relu, sim_relu)
# Plots.savefig(fig, "ACC-ReLU.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

fig = plot_helper(sol_tanh, sim_tanh)
fig = DisplayAs.Text(DisplayAs.PNG(fig))
# savefig(fig, "ACC-tanh.png")  # command to save the plot to a file

end
nothing
