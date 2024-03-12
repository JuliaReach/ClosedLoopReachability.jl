module TORA

using ClosedLoopReachability
import DifferentialEquations, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: UniformAdditivePostprocessing, NoSplitter, LinearMapPostprocessing
using Plots: plot, plot!, lens!, bbox

const verification = false;

vars_idx = Dict(:states => 1:4, :controls => 5)

@taylorize function TORA!(dx, x, p, t)
    x₁, x₂, x₃, x₄, u = x

    dx[1] = x₂
    dx[2] = -x₁ + (0.1 * sin(x₃))
    dx[3] = x₄
    dx[4] = u
    dx[5] = zero(u)
    return dx
end;

path = @current_path("TORA", "TORA_ReLU_controller.polar")
controller_ReLU = read_POLAR(path)

control_postprocessing1 = UniformAdditivePostprocessing(-10.0)

period1 = 1.0;

path = @current_path("TORA", "TORA_ReLUtanh_controller.polar")
controller_relutanh = read_POLAR(path)
path = @current_path("TORA", "TORA_sigmoid_controller.polar")
controller_sigmoid = read_POLAR(path)

control_postprocessing2 = LinearMapPostprocessing(11.0)

period2 = 0.5;

X₀1 = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])
U = ZeroSet(1);

ivp1 = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀1 × U);

safe_states = cartesian_product(BallInf(zeros(4), 2.0), Universe(1))

predicate1(sol) = overapproximate(sol, Hyperrectangle) ⊆ safe_states

T1 = 20.0  # time horizon
T1_warmup = 2 * period1  # shorter time horizon for warm-up run
T1_reach = verification ? T1 : T1_warmup;  # shorter time horizon if not verifying

X₀2 = Hyperrectangle(low=[-0.77, -0.45, 0.51, -0.3], high=[-0.75, -0.43, 0.54, -0.28])
U = ZeroSet(1);

ivp2 = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀2 × U);

goal_states = cartesian_product(Hyperrectangle(low=[-0.1, -0.9], high=[0.2, -0.6]),
                                Universe(3))

predicate_set2(R) = overapproximate(R, Hyperrectangle) ⊆ goal_states

predicate2(sol) = predicate_set2(sol[end][end])

T2 = 5.0  # time horizon
T2_warmup = 2 * period2;  # shorter time horizon for warm-up run

algorithm_plant = TMJets(abstol=1e-10, orderT=8, orderQ=3);

algorithm_controller = DeepZ();

function benchmark(prob; T=T, splitter, predicate, silent::Bool=false)
    # Solve the controlled system:
    silent || println("Flowpipe construction:")
    res = @timed solve(prob; T=T, algorithm_controller=algorithm_controller,
                       algorithm_plant=algorithm_plant, splitter=splitter)
    sol = res.value
    silent || print_timed(res)

    # Check the property:
    silent || println("Property checking:")
    res = @timed predicate(sol)
    silent || print_timed(res)
    if res.value
        silent || println("  The property is satisfied.")
    else
        silent || println("  The property may be violated.")
    end

    return sol
end;

function run(; scenario1::Bool, ReLUtanh_activations)
    if scenario1
        println("# Running analysis of scenario 1 with ReLU activations")
        prob = ControlledPlant(ivp1, controller_ReLU, vars_idx, period1;
                               postprocessing=control_postprocessing1)
        splitter = verification ? BoxSplitter([4, 4, 3, 5]) : NoSplitter()
        predicate = predicate1
        T = T1_reach
        T_warmup = T1_warmup
    elseif ReLUtanh_activations
        println("# Running analysis of scenario 2 with ReLUtanh activations")
        prob = ControlledPlant(ivp2, controller_relutanh, vars_idx, period2;
                               postprocessing=control_postprocessing2)
        splitter = NoSplitter()
        predicate = predicate2
        T = T2
        T_warmup = T2_warmup
    else
        println("# Running analysis of scenario 2 with sigmoid activations")
        prob = ControlledPlant(ivp2, controller_sigmoid, vars_idx, period2;
                               postprocessing=control_postprocessing2)
        splitter = NoSplitter()
        predicate = predicate2
        T = T2
        T_warmup = T2_warmup
    end

    # Run the verification benchmark:
    benchmark(prob; T=T_warmup, splitter=splitter, predicate=predicate, silent=true)  # warm-up
    res = @timed benchmark(prob; T=T, splitter=splitter, predicate=predicate)  # benchmark
    sol = res.value
    println("Total analysis time:")
    print_timed(res)

    # Compute some simulations:
    println("Simulation:")
    if scenario1
        res = @timed simulate(prob; T=T, trajectories=10, include_vertices=true)
    else
        res = @timed simulate(prob; T=T, trajectories=1, include_vertices=true)
    end
    sim = res.value
    print_timed(res)

    return sol, sim
end;

sol_r, sim_r = run(scenario1=true, ReLUtanh_activations=nothing);

sol_rt, sim_rt = run(scenario1=false, ReLUtanh_activations=true);

sol_sig, sim_sig = run(scenario1=false, ReLUtanh_activations=false);

solz = overapproximate(sol_r, Zonotope);

function plot_helper1!(fig, vars)
    plot!(fig, project(safe_states, vars); color=:lightgreen, lab="safe")
    plot!(fig, solz; vars=vars, color=:yellow, lab="")
    plot!(fig, project(X₀1, vars); c=:cornflowerblue, alpha=1, lab="X₀")
    plot_simulation!(fig, sim_r; vars=vars, color=:black, lab="")
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end;

vars = (1, 2)
fig = plot(xlab="x₁", ylab="x₂")
fig = plot_helper1!(fig, vars)
# savefig("TORA-ReLU-x1-x2.png")  # command to save the plot to a file

vars = (3, 4)
fig = plot(xlab="x₃", ylab="x₄")
fig = plot_helper1!(fig, vars)
# savefig("TORA-ReLU-x3-x4.png")  # command to save the plot to a file

function plot_helper2(sol, sim)
    vars = (1, 2)
    fig = plot(xlab="x₁", ylab="x₂")
    plot!(fig, project(goal_states, vars); color=:cyan, lab="goal")
    plot!(fig, sol; vars=vars, color=:yellow, lab="")
    plot!(fig, project(X₀2, vars); c=:cornflowerblue, alpha=1, lab="X₀")
    plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    return fig
end;

fig = plot_helper2(sol_rt, sim_rt)
lens!(fig, [-0.785, -0.735], [-0.47, -0.41]; inset=(1, bbox(0.2, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[-0.77, -0.75], yticks=[-0.45, -0.43], subplot=3)
lens!(fig, [0.0, 0.25], [-0.85, -0.7]; inset=(1, bbox(0.6, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[0, 0.2], yticks=[-0.8, -0.7], subplot=3)
fig = DisplayAs.Text(DisplayAs.PNG(fig))
# savefig("TORA-ReLUtanh-x1-x2.png")  # command to save the plot to a file

fig = plot_helper2(sol_sig, sim_sig)
lens!(fig, [-0.785, -0.735], [-0.47, -0.41]; inset=(1, bbox(0.2, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[-0.77, -0.75], yticks=[-0.45, -0.43], subplot=3)
lens!(fig, [0.09, 0.22], [-0.9, -0.8]; inset=(1, bbox(0.6, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[0.1, 0.2], yticks=[-0.9, -0.8], subplot=3)
fig = DisplayAs.Text(DisplayAs.PNG(fig))
# savefig("TORA-sigmoid-x1-x2.png")  # command to save the plot to a file

end
nothing
