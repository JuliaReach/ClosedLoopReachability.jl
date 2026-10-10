module TORA

using ClosedLoopReachability
import OrdinaryDiffEq, Plots, DisplayAs
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

X₀1 = Hyperrectangle([0.65, -0.65, -0.35, 0.55], 0.01 * [0.05, 0.05, 0.05, 0.05])
U = ZeroSet(1);

ivp1 = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀1 × U);

safe_states = cartesian_product(BallInf(zeros(4), 2.0), Universe(1))

predicate1(sol, T) = (overapproximate(sol, Hyperrectangle) ⊆ safe_states, nothing)

T1 = 20.0  # time horizon
T1_warmup = 2 * period1  # shorter time horizon for warm-up run

c = [-0.76, -0.44, 0.525, -0.29]
r = [0.01, 0.01, 0.015, 0.01]
X₀2 = verification ? Hyperrectangle(c, r) : Hyperrectangle(c, 0.01 * r)
U = ZeroSet(1);

ivp2 = @ivp(x' = TORA!(x), dim: 5, x(0) ∈ X₀2 × U);

goal_states = cartesian_product(Hyperrectangle(low=[-0.1, -0.9], high=[0.2, -0.6]),
                                Universe(3))

predicate_set2(R, t) = overapproximate(R, Hyperrectangle, t) ⊆ goal_states

function predicate2(sol, T)
    times = Float64[]
    for F in sol
        if T ∉ tspan(F)
            continue
        end
        R = F[end]
        t = tstart(R)
        steps = 10
        Δt = (tend(R) - tstart(R)) / steps
        satisfied = false
        for j in 0:steps
            if j == steps
                t = tend(R)  # needed for rounding issues
            end
            if predicate_set2(R, t)
                satisfied = true
                push!(times, t)
                break
            end
            t += Δt
        end
        if !satisfied
            return false, times
        end
    end
    return true, times
end

T2 = 5.0  # time horizon
T2_warmup = 2 * period2;  # shorter time horizon for warm-up run

algorithm_plant = TMJets(abstol=1e-3, orderT=3, orderQ=2);

algorithm_controller = DeepZ();

function benchmark(prob; T, splitter, algorithm_plant, predicate,
                   silent::Bool=false)
    # Solve the controlled system:
    silent || println("Flowpipe construction:")
    res = @timed solve(prob; T=T, algorithm_controller=algorithm_controller,
                       algorithm_plant=algorithm_plant, splitter=splitter)
    sol = res.value
    silent || print_timed(res)

    # Check the property:
    silent || println("Property checking:")
    res = @timed predicate(sol, T)
    silent || print_timed(res)
    if res.value[1]
        silent || println("  The property is satisfied.")
        result = "verified"
    else
        silent || println("  The property may be violated.")
        result = "not verified"
    end

    return sol, result, res.value[2]
end;

function run(; scenario1::Bool, ReLUtanh_activations)
    if scenario1
        println("# Running analysis of scenario 1 with ReLU activations")
        prob = ControlledPlant(ivp1, controller_ReLU, vars_idx, period1;
                               postprocessing=control_postprocessing1)
        splitter = NoSplitter()
        predicate = predicate1
        T = T1
        T_warmup = T1_warmup
    else
        if ReLUtanh_activations
            println("# Running analysis of scenario 2 with ReLUtanh activations")
            controller = controller_relutanh
            splitter = verification ?
                       BoxSplitter([[-0.763, -0.757], [-0.445, -0.44, -0.435], [0.52], [-0.29]]) :
                       NoSplitter()
        else
            println("# Running analysis of scenario 2 with sigmoid activations")
            controller = controller_sigmoid
            splitter = verification ?
                       BoxSplitter([[-0.768, -0.766, -0.764, -0.762, -0.76, -0.758, -0.755, -0.752], [-0.449, -0.447, -0.445, -0.443, -0.441, -0.439, -0.437, -0.435, -0.433, -0.431], [0.518, 0.525, 0.532], [-0.2934, -0.2867]]) :
                       NoSplitter()
        end
        prob = ControlledPlant(ivp2, controller, vars_idx, period2;
                               postprocessing=control_postprocessing2)
        predicate = predicate2
        T = T2
        T_warmup = T2_warmup
    end

    # Run the verification benchmark:
    benchmark(prob; T=T_warmup, splitter=splitter,
        algorithm_plant=algorithm_plant, predicate=predicate, silent=true)  # warm-up
    res = @timed benchmark(prob; T=T, splitter=splitter,
        algorithm_plant=algorithm_plant, predicate=predicate)  # benchmark
    sol, result, times = res.value
    @assert (result == "verified") "verification failed"
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

    return sol, sim, times
end;

sol_r, sim_r, _ = run(scenario1=true, ReLUtanh_activations=nothing);

sol_sig, sim_sig, times_sig = run(scenario1=false, ReLUtanh_activations=false);

sol_rt, sim_rt, times_rt = run(scenario1=false, ReLUtanh_activations=true);

solz = overapproximate(sol_r, Zonotope);

function plot_helper1(vars)
    fig = plot()
    plot!(fig, project(safe_states, vars); color=:lightgreen, lab="safe")
    plot!(fig, solz; vars=vars, color=:yellow, lw=0, alpha=1, lab="")
    plot!(fig, project(X₀1, vars); c=:cornflowerblue, alpha=1, lab="X₀")
    plot_simulation!(fig, sim_r; vars=vars, color=:black, lab="")
    return fig
end;

vars = (1, 2)
fig = plot_helper1(vars)
plot!(fig; xlab="x₁", ylab="x₂")
# Plots.savefig(fig, "TORA-ReLU-x1-x2.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

vars = (3, 4)
fig = plot_helper1(vars)
plot!(fig; xlab="x₃", ylab="x₄")
# Plots.savefig(fig, "TORA-ReLU-x3-x4.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

Tint = try convert(Int, T2) catch; T2 end;

function plot_helper2(sol, sim, times)
    vars = (1, 2)
    fig = plot()
    plot!(fig, project(goal_states, vars); color=:cyan, lab="goal")
    plot!(fig, sol; vars=vars, color=:yellow, lw=0, alpha=1, lab="")
    lab = "reach set at t ≈ $Tint"
    i = 1
    for F in sol
        if T2 ∉ tspan(F)
            continue
        end
        plot!(fig, overapproximate(F[end], Zonotope, times[i]);
              vars=vars, color=:orange, lab=lab)
        lab = ""
        i += 1
    end
    plot!(fig, project(X₀2, vars); c=:cornflowerblue, alpha=1, lab="X₀")
    plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    plot!(fig; xlab="x₁", ylab="x₂")
    return fig
end;

fig = plot_helper2(sol_sig, sim_sig, times_sig)
lens!(fig, [-0.785, -0.735], [-0.47, -0.41]; inset=(1, bbox(0.2, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[-0.77, -0.75], yticks=[-0.45, -0.43], subplot=3)
lens!(fig, [0.09, 0.22], [-0.9, -0.8]; inset=(1, bbox(0.6, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[0.1, 0.2], yticks=[-0.9, -0.8], subplot=3)
# Plots.savefig(fig, "TORA-sigmoid.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

fig = plot_helper2(sol_rt, sim_rt, times_rt)
lens!(fig, [-0.785, -0.735], [-0.47, -0.41]; inset=(1, bbox(0.2, 0.4, 0.2, 0.2)),
      lc=:black, xticks=[-0.77, -0.75], yticks=[-0.45, -0.43], subplot=3)
if !verification
    lens!(fig, [0.05, 0.22], [-0.92, -0.7]; inset=(1, bbox(0.6, 0.4, 0.15, 0.2)),
          lc=:black, xticks=[0, 0.2], yticks=[-0.8, -0.7], subplot=3)
end
# Plots.savefig(fig, "TORA-ReLUtanh.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

end
nothing
