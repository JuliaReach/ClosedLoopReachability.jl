module InvertedTwoLinkPendulum

using ClosedLoopReachability
import DifferentialEquations, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: Specification
using Plots: plot, plot!, xlims!, ylims!

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
    if falsification
        # Choose a single point in the initial states (here: the top-most one):
        if less_robust_scenario
            X₀ = Singleton(high(X₀))
        else
            X₀ = Singleton(low(X₀))
        end
    end
    U₀ = ZeroSet(2)

    period = less_robust_scenario ? period_lr : period_mr

    # The control problem is:
    ivp = @ivp(x' = InvertedTwoLinkPendulum!(x), dim: 6, x(0) ∈ X₀ × U₀)
    prob = ControlledPlant(ivp, controller, vars_idx, period)

    # Safety specification:
    if less_robust_scenario
        box = BallInf(fill(0.35, 4), 1.35)
    else
        box = BallInf(fill(0.5, 4), 1.0)
    end
    safe_states = cartesian_product(box, Universe(2))

    predicate_set(R) = isdisjoint(overapproximate(R, Hyperrectangle), safe_states)

    function predicate(sol; silent::Bool=false)
        for F in sol, R in F
            if predicate_set(R)
                silent || println("  Violation for time range $(tspan(R)).")
                return true
            end
        end
        return false
    end

    if falsification
        # Falsification can run for a shorter time horizon:
        if less_robust_scenario
            k = 5
        else
            k = 7
        end
    else
        k = 20
    end
    T = k * period  # time horizon

    spec = Specification(T, predicate, safe_states)

    return prob, spec
end;

algorithm_plant = TMJets(abstol=1e-9, orderT=8, orderQ=1);

algorithm_controller = DeepZ();

function benchmark(prob, spec; T, silent::Bool=false)
    # Solve the controlled system:
    silent || println("Flowpipe construction:")
    res = @timed solve(prob; T=T, algorithm_controller=algorithm_controller,
                       algorithm_plant=algorithm_plant)
    sol = res.value
    silent || print_timed(res)

    # Check the property:
    silent || println("Property checking:")
    res = @timed spec.predicate(sol; silent=silent)
    silent || print_timed(res)
    if res.value
        silent || println("  The property is violated.")
    else
        silent || println("  The property may be satisfied.")
    end

    return sol
end

function run(; less_robust_scenario::Bool)
    if less_robust_scenario
        println("# Running analysis with less robust scenario")
        T_warmup = 2 * period_lr  # shorter time horizon for warm-up run
    else
        println("# Running analysis with more robust scenario")
        T_warmup = 2 * period_mr  # shorter time horizon for warm-up run
    end
    prob, spec = InvertedTwoLinkPendulum_spec(less_robust_scenario)

    # Run the verification benchmark:
    benchmark(prob, spec; T=T_warmup, silent=true)  # warm-up
    res = @timed benchmark(prob, spec; T=spec.T)  # benchmark
    sol = res.value
    println("total analysis time")
    print_timed(res)

    # Compute some simulations:
    println("simulation")
    trajectories = falsification ? 1 : 10
    res = @timed simulate(prob; T=spec.T, trajectories=trajectories,
                          include_vertices=!falsification)
    sim = res.value
    print_timed(res)

    return sol, sim, prob, spec
end;

sol_lr, sim_lr, prob_lr, spec_lr = run(less_robust_scenario=true);

sol_mr, sim_mr, prob_mr, spec_mr = run(less_robust_scenario=false);

function plot_helper!(fig, vars, sol, sim, prob, spec, scenario)
    safe_states = spec.ext
    plot!(fig, project(safe_states, vars); color=:lightgreen, lab="safe")
    plot!(fig, sol; vars=vars, color=:yellow, lab="")
    plot!(fig, project(initial_state(prob), vars); c=:cornflowerblue, alpha=1, lab="X₀")
    lab_sim = falsification ? "simulation" : ""
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
    if falsification
        plot!(leg=:topleft)
    end
    # Command to save the plot to a file:
    # savefig("InvertedTwoLinkPendulum-$scenario-x$(vars[1])-x$(vars[2]).png")
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end;

vars=(3, 4)
fig = plot(xlab="θ₁'", ylab="θ₂'")
xlims!(-0.7, 1.7)
ylims!(-1.6, 1.5)
fig = plot_helper!(fig, vars, sol_lr, sim_lr, prob_lr, spec_lr, "less-robust")

vars=(3, 4)
fig = plot(xlab="θ₁'", ylab="θ₂'")
if falsification
    ylims!(-1.0, 1.5)
else
    xlims!(-1.8, 1.5)
    ylims!(-1.6, 1.5)
end
fig = plot_helper!(fig, vars, sol_mr, sim_mr, prob_mr, spec_mr, "more-robust")

end
nothing
