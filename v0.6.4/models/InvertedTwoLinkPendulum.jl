module InvertedTwoLinkPendulum

using ClosedLoopReachability
import OrdinaryDiffEq, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: Specification, NoSplitter
using Plots: plot, plot!, xlims!, ylims!

const verification = false;

const falsification = true;

vars_idx = Dict(:states => 1:4, :controls => 5:6)

const m = 0.5
const L = 0.5
const c = 0.0
const g = 1.0
const gL = g / L
const mL = 1 / (m * L^2)

@taylorize function InvertedTwoLinkPendulum!(dx, x, p, t)
    θ₁, θ₂, θ₁′, θ₂′, T₁, T₂ = x

    Δ12 = θ₁ - θ₂
    cos12 = cos(Δ12)
    x3sin12 = θ₁′^2 * sin(Δ12)
    x4sin12 = θ₂′^2 * sin(Δ12) / 2
    gLsin1 = gL * sin(θ₁)
    gLsin2 = gL * sin(θ₂)
    T1_frac = (T₁ - c * θ₁′) * (0.5 * mL)
    T2_frac = (T₂ - c * θ₂′) * mL
    bignum = x3sin12 - cos12 * (gLsin1 - x4sin12 + T1_frac) + gLsin2 + T2_frac
    denom = cos12^2 / 2 - 1

    dx[1] = θ₁′
    dx[2] = θ₂′
    dx[3] = cos12 * bignum / (2 * denom) - x4sin12 + gLsin1 + T1_frac
    dx[4] = -bignum / denom
    dx[5] = zero(T₁)
    dx[6] = zero(T₂)
    return dx
end;

path = @current_path("InvertedTwoLinkPendulum",
                     "InvertedTwoLinkPendulum_controller_less_robust.polar")
controller_lr = read_POLAR(path)

path = @current_path("InvertedTwoLinkPendulum",
                     "InvertedTwoLinkPendulum_controller_more_robust.polar")
controller_mr = read_POLAR(path);

period_lr = 0.05
period_mr = 0.02;

function InvertedTwoLinkPendulum_spec(less_robust_scenario::Bool)
    controller = less_robust_scenario ? controller_lr : controller_mr

    X₀ = BallInf(fill(1.15, 4), 0.15)
    if falsification && !less_robust_scenario
        # Choose a single point in the initial states (here: the top-most one):
        X₀ = Singleton(high(X₀))
    end
    U₀ = ZeroSet(2)

    period = less_robust_scenario ? period_lr : period_mr

    # The control problem is:
    ivp = @ivp(x' = InvertedTwoLinkPendulum!(x), dim: 6, x(0) ∈ X₀ × U₀)
    prob = ControlledPlant(ivp, controller, vars_idx, period)

    # Safety specification:
    if less_robust_scenario
        box = BallInf(fill(0.15, 4), 1.85)
    else
        box = BallInf(fill(0.0, 4), 1.5)
    end
    safe_states = cartesian_product(box, Universe(2))

    predicate_set_safe(R) = overapproximate(R, Hyperrectangle) ⊆ safe_states
    predicate_set_unsafe(R) = isdisjoint(overapproximate(R, Hyperrectangle), safe_states)

    function predicate_safe(sol; silent::Bool=false)
        for F in sol, R in F
            if !predicate_set_safe(R)
                silent || println("  Potential violation for time range $(tspan(R)).")
                return false
            end
        end
        return true
    end

    function predicate_unsafe(sol)
        for F in sol, R in F
            if predicate_set_unsafe(R)
                return true
            end
        end
        return false
    end

    if less_robust_scenario
        predicate = predicate_safe
    else
        predicate = predicate_unsafe
    end

    if !verification && less_robust_scenario
        # Run for a shorter time horizon if verification is deactivated:
        k = 2
    elseif falsification && !less_robust_scenario
        # Falsification can run for a shorter time horizon:
        k = 18
    else
        k = 20
    end
    T = k * period  # time horizon

    spec = Specification(T, predicate, safe_states)

    return prob, spec
end;

algorithm_controller = DeepZ();

function benchmark(prob, spec; T, algorithm_plant, splitter, less_robust_scenario, silent::Bool=false)
    # Solve the controlled system:
    silent || println("Flowpipe construction:")
    res = @timed solve(prob; T=T, algorithm_controller=algorithm_controller,
                       algorithm_plant=algorithm_plant, splitter=splitter)
    sol = res.value
    silent || print_timed(res)

    # Check the property:
    silent || println("Property checking:")
    if less_robust_scenario
        res = @timed spec.predicate(sol; silent=silent)
        silent || print_timed(res)
        if res.value
            silent || println("  The property is verified.")
            result = "verified"
        else
            silent || println("  The property may be violated.")
            result = "not verified"
        end
    else
        res = @timed spec.predicate(sol)
        silent || print_timed(res)
        if res.value
            silent || println("  The property is violated.")
            result = "falsified"
        else
            silent || println("  The property may be satisfied.")
            result = "not falsified"
        end
    end

    return sol, result
end

function run(; less_robust_scenario::Bool)
    if less_robust_scenario
        println("# Running analysis with less robust scenario")
        algorithm_plant = TMJets(abstol=1e-9, orderT=5, orderQ=1)
        splitter = !verification ? BoxSplitter([[1.15], [1.15], Float64[], [1.2]]) :
            BoxSplitter([[1.15], [1.15], [1.12, 1.25], [1.05, 1.11, 1.165, 1.21, 1.257]])
        T_warmup = 2 * period_lr  # shorter time horizon for warm-up run
    else
        println("# Running analysis with more robust scenario")
        algorithm_plant = TMJets(abstol=1e-2, orderT=3, orderQ=1)
        splitter = NoSplitter()
        T_warmup = 2 * period_mr  # shorter time horizon for warm-up run
    end
    prob, spec = InvertedTwoLinkPendulum_spec(less_robust_scenario)

    # Run the verification/falsification benchmark:
    benchmark(prob, spec; T=T_warmup, algorithm_plant=algorithm_plant, splitter=splitter,
              less_robust_scenario=less_robust_scenario, silent=true)  # warm-up
    res = @timed benchmark(prob, spec; T=spec.T, algorithm_plant=algorithm_plant,  # benchmark
                           splitter=splitter, less_robust_scenario=less_robust_scenario)
    sol, result = res.value
    if verification && less_robust_scenario
        @assert (result == "verified") "verification failed"
    elseif !less_robust_scenario
        @assert (result == "falsified") "falsification failed"
    end
    println("Total analysis time:")
    print_timed(res)

    # Compute some simulations:
    println("Simulation:")
    simulations = less_robust_scenario || !falsification
    trajectories = simulations ? 10 : 1
    res = @timed simulate(prob; T=spec.T, trajectories=trajectories,
                          include_vertices=simulations)
    sim = res.value
    print_timed(res)

    return sol, sim, prob, spec
end;

sol_lr, sim_lr, prob_lr, spec_lr = run(less_robust_scenario=true);

sol_mr, sim_mr, prob_mr, spec_mr = run(less_robust_scenario=false);

function plot_helper(vars, sol, sim, prob, spec; lab_sim="")
    safe_states = spec.ext
    fig = plot()
    plot!(fig, project(safe_states, vars); color=:lightgreen, lab="safe")
    plot!(fig, sol; vars=vars, color=:yellow, lw=0, alpha=1, lab="")
    plot!(fig, project(initial_state(prob), vars); c=:cornflowerblue, alpha=1, lab="X₀")
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
    return fig
end;

vars = (1, 2)
fig = plot_helper(vars, sol_lr, sim_lr, prob_lr, spec_lr)
plot!(fig; xlab="θ₁", ylab="θ₂")
# Command to save the plot to a file:
# Plots.savefig(fig, "InvertedTwoLinkPendulum-less-robust-x1-x2.png")
fig = DisplayAs.Text(DisplayAs.PNG(fig))

vars = (3, 4)
fig = plot_helper(vars, sol_lr, sim_lr, prob_lr, spec_lr)
plot!(fig; xlab="θ₁'", ylab="θ₂'")
# Command to save the plot to a file:
# Plots.savefig(fig, "InvertedTwoLinkPendulum-less-robust-x3-x4.png")
fig = DisplayAs.Text(DisplayAs.PNG(fig))

vars = (3, 4)
lab_sim = falsification ? "simulation" : ""
fig = plot_helper(vars, sol_mr, sim_mr, prob_mr, spec_mr; lab_sim=lab_sim)
plot!(fig; xlab="θ₁'", ylab="θ₂'")
# Command to save the plot to a file:
# Plots.savefig(fig, "InvertedTwoLinkPendulum-more-robust.png")
fig = DisplayAs.Text(DisplayAs.PNG(fig))

end
nothing
