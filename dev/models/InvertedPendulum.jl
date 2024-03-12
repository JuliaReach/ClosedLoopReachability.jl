module InvertedPendulum

using ClosedLoopReachability
import DifferentialEquations, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: SingleEntryVector
using Plots: plot, plot!, xlims!, ylims!

const falsification = true;

vars_idx = Dict(:states => 1:2, :controls => 3)

const m = 0.5
const L = 0.5
const c = 0.0
const g = 1.0
const gL = g / L
const mL = 1 / (m * L^2)

@taylorize function InvertedPendulum!(dx, x, p, t)
    θ, θ′, T = x

    dx[1] = θ′
    dx[2] = gL * sin(θ) + mL * (T - c * θ′)
    dx[3] = zero(T)
    return dx
end;

path = @current_path("InvertedPendulum", "InvertedPendulum_controller.polar")
controller = read_POLAR(path);

period = 0.05;

X₀ = BallInf([1.1, 0.1], 0.1)
if falsification
    # Choose a single point in the initial states (here: the top-most one):
    X₀ = Singleton(high(X₀))
end
U₀ = ZeroSet(1);

ivp = @ivp(x' = InvertedPendulum!(x), dim: 3, x(0) ∈ X₀ × U₀)
prob = ControlledPlant(ivp, controller, vars_idx, period);

unsafe_states = HalfSpace(SingleEntryVector(1, 3, -1.0), -1.0)

function predicate_set(R)
    t = tspan(R)
    return t.lo >= 0.5 && t.hi <= 1.0 &&
           overapproximate(R, Hyperrectangle) ⊆ unsafe_states
end

function predicate(sol; silent::Bool=false)
    for F in sol
        t = tspan(F)
        if t.hi < 0.5 || t.lo > 1.0
            continue
        end
        for R in F
            if predicate_set(R)
                silent || println("  Violation for time range $(tspan(R)).")
                return true
            end
        end
    end
    return false
end

if falsification
    k = 11  # falsification can run for a shorter time horizon
else
    k = 20
end
T = k * period
T_warmup = 2 * period;  # shorter time horizon for warm-up run

algorithm_plant = TMJets(abstol=1e-7, orderT=4, orderQ=1);

algorithm_controller = DeepZ();

function benchmark(; T=T, silent::Bool=false)
    # Solve the controlled system:
    silent || println("Flowpipe construction:")
    res = @timed solve(prob; T=T, algorithm_controller=algorithm_controller,
                       algorithm_plant=algorithm_plant)
    sol = res.value
    silent || print_timed(res)

    # Check the property:
    silent || println("Property checking:")
    res = @timed predicate(sol; silent=silent)
    silent || print_timed(res)
    if res.value
        silent || println("  The property is violated.")
    else
        silent || println("  The property may be satisfied.")
    end

    return sol
end;

benchmark(T=T_warmup, silent=true)  # warm-up
res = @timed benchmark(T=T)  # benchmark
sol = res.value
println("Total analysis time:")
print_timed(res)

println("Simulation:")
res = @timed simulate(prob; T=T, trajectories=falsification ? 1 : 10,
                      include_vertices=!falsification)
sim = res.value
print_timed(res);

function plot_helper()
    vars = (0, 1)
    fig = plot(ylab="θ")
    unsafe_states_projected = cartesian_product(Interval(0.5, 1.0),
                                                project(unsafe_states, [vars[2]]))
    plot!(fig, unsafe_states_projected; color=:red, alpha=:0.2, lab="unsafe")
    plot!(fig, sol; vars=vars, color=:yellow, lab="")
    initial_states_projected = cartesian_product(Singleton([0.0]), project(X₀, [vars[2]]))
    plot!(fig, initial_states_projected; c=:cornflowerblue, alpha=1, lab="X₀")
    if falsification
        xlims!(0, T)
        ylims!(0.95, 1.22)
    else
        xlims!(0, T)
        ylims!(0.55, 1.3)
    end
    lab_sim = falsification ? "simulation" : ""
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
    fig = DisplayAs.Text(DisplayAs.PNG(fig))
end;

fig = plot_helper()
# savefig("InvertedPendulum.png")  # command to save the plot to a file

end
nothing
