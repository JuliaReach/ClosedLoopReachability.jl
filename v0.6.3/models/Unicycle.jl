module Unicycle

using ClosedLoopReachability
import OrdinaryDiffEq, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: UniformAdditivePostprocessing
using Plots: plot, plot!, lens!, bbox

vars_idx = Dict(:states => 1:4, :disturbances => [5], :controls => 6:7)

@taylorize function Unicycle!(dx, x, p, t)
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

path = @current_path("Unicycle", "Unicycle_controller.polar")
controller = read_POLAR(path)

control_postprocessing = UniformAdditivePostprocessing(-20.0);

period = 0.2;

X₀ = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5, -1e-4],
                    high=[9.55, -4.45, 2.11, 1.51, 1e-4])
U₀ = ZeroSet(2);

ivp = @ivp(x' = Unicycle!(x), dim: 7, x(0) ∈ X₀ × U₀)
prob = ControlledPlant(ivp, controller, vars_idx, period;
                       postprocessing=control_postprocessing);

goal_set = cartesian_product(Hyperrectangle(zeros(4), [0.6, 0.2, 0.06, 0.3]),
                             Universe(3))

predicate_set(R) = overapproximate(R, Hyperrectangle, tend(R)) ⊆ goal_set

predicate(sol) = all(predicate_set(F[end]) for F in sol if T ∈ tspan(F))

T = 10.0
T_warmup = 2 * period;  # shorter time horizon for warm-up run

algorithm_plant = TMJets(abstol=1e-1, orderT=3, orderQ=1);

algorithm_controller = DeepZ()
splitter = BoxSplitter([3, 1, 7, 1]);

function benchmark(; T=T, silent::Bool=false)
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
        result = "verified"
    else
        silent || println("  The property may be violated.")
        result = "not verified"
    end

    return sol, result
end;

benchmark(T=T_warmup, silent=true)  # warm-up
res = @timed benchmark(T=T)  # benchmark
sol, result = res.value
@assert (result == "verified") "verification failed"
println("Total analysis time:")
print_timed(res)

println("Simulation:")
res = @timed simulate(prob; T=T, trajectories=5, include_vertices=false)
sim = res.value
print_timed(res);

solz = overapproximate(sol, Zonotope)
Tint = try convert(Int, T) catch; T end;

function plot_helper(vars; show_simulation::Bool=true)
    fig = plot()
    plot!(fig, project(goal_set, vars); color=:cyan, alpha=0.5, lab="goal")
    plot!(fig, solz; vars=vars, color=:yellow, lw=0, alpha=1, lab="")
    plot!(fig, project(X₀, vars); color=:cornflowerblue, alpha=1, lab="X₀")
    lab = "reach set at t = $Tint"
    for F in sol
        if T ∈ tspan(F)
            plot!(fig, overapproximate(F[end], Zonotope, tend(F));
                  vars=vars, color=:orange, lab=lab)
            lab = ""
        end
    end
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
    return fig
end;

vars = (1, 2)
fig = plot_helper(vars)
plot!(fig; xlab="x₁", ylab="x₂", leg=:bottomleft)
lens!(fig, [9.49, 9.56], [-4.51, -4.44]; inset=(1, bbox(0.65, 0.05, 0.25, 0.25)),
      lc=:black, xticks=[9.5, 9.55], yticks=[-4.5, -4.45], subplot=2)
lens!(fig, [0.3, 0.7], [-0.25, 0.25]; inset=(1, bbox(0.1, 0.3, 0.25, 0.25)),
      lc=:black, xticks=[0.4, 0.6], yticks=[-0.2, 0.2], subplot=3)
# Plots.savefig(fig, "Unicycle-x1-x2.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

vars = (3, 4)
fig = plot_helper(vars; show_simulation=false)
plot!(fig; xlab="x₃", ylab="x₄", leg=:bottom)
lens!(fig, [2.09, 2.12], [1.495, 1.515]; inset=(1, bbox(0.72, 0.54, 0.25, 0.25)),
      lc=:black, xticks=[2.1, 2.11], yticks=[1.5, 1.51], subplot=2)
lens!(fig, [-0.1, 0.03], [-0.4, -0.15]; inset=(1, bbox(0.1, 0.1, 0.25, 0.25)),
      lc=:black, xticks=[-0.08, 0], yticks=[-0.3, -0.2], subplot=3)
# Plots.savefig(fig, "Unicycle-x3-x4.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

end
nothing
