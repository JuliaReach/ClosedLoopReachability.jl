module Airplane

using ClosedLoopReachability
import OrdinaryDiffEq, Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using Plots: plot, plot!, xlims!, ylims!

const falsification = true;

vars_idx = Dict(:states => 1:12, :controls => 13:18)

const m = 1.0
const g = 1.0

Tψ(ψ) = [ cos(ψ)  -sin(ψ)  zero(ψ);
          sin(ψ)   cos(ψ)  zero(ψ);
         zero(ψ)  zero(ψ)   one(ψ)]

Tθ(θ) = [ cos(θ)  zero(θ)   sin(θ);
         zero(θ)   one(θ)  zero(θ);
         -sin(θ)  zero(θ)   cos(θ)]

Tϕ(ϕ) = [one(ϕ)  zero(ϕ)  zero(ϕ);
         zero(ϕ)  cos(ϕ)  -sin(ϕ);
         zero(ϕ)  sin(ϕ)   cos(ϕ)]

Rϕθ(ϕ, θ) = [ one(ϕ)  tan(θ) * sin(ϕ)  tan(θ) * cos(ϕ);
             zero(ϕ)           cos(θ)          -sin(ϕ);
             zero(ϕ)  sec(θ) * sin(ϕ)  sec(θ) * cos(ϕ)]

@taylorize function Airplane!(dx, x, params, t)
    s_x, s_y, s_z, v_x, v_y, v_z, ϕ, θ, ψ, r, p, q, Fx, Fy, Fz, Mx, My, Mz = x

    T_ψ = Tψ(ψ)
    T_θ = Tθ(θ)
    T_ϕ = Tϕ(ϕ)
    mat_1 = T_ψ * T_θ * T_ϕ
    xyz = mat_1 * vcat(v_x, v_y, v_z)

    mat_2 = Rϕθ(ϕ, θ)
    ϕθψ = mat_2 * vcat(p, q, r)

    dx[1] = xyz[1]
    dx[2] = xyz[2]
    dx[3] = xyz[3]
    dx[4] = -g * sin(θ) + Fx / m - q * v_z + r * v_y
    dx[5] = g * cos(θ) * sin(ϕ) + Fy / m - r * v_x + p * v_z
    dx[6] = g * cos(θ) * cos(ϕ) + Fz / m - p * v_y + q * v_x
    dx[7] = ϕθψ[1]
    dx[8] = ϕθψ[2]
    dx[9] = ϕθψ[3]
    dx[10] = Mz  # simplified term
    dx[11] = Mx  # simplified term
    dx[12] = My  # simplified term
    dx[13] = zero(Fx)
    dx[14] = zero(Fy)
    dx[15] = zero(Fz)
    dx[16] = zero(Mx)
    dx[17] = zero(My)
    dx[18] = zero(Mz)
    return dx
end;

path = @current_path("Airplane", "Airplane_controller.polar")
controller = read_POLAR(path);

period = 0.1;

X₀ = Hyperrectangle(low=[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    high=[0.0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
if falsification
    # Choose a single point in the initial states (here: the top-most one):
    X₀ = Singleton(high(X₀))
end
U₀ = ZeroSet(6);

ivp = @ivp(x' = Airplane!(x), dim: 18, x(0) ∈ X₀ × U₀)
prob = ControlledPlant(ivp, controller, vars_idx, period);

safe_states = concretize(CartesianProductArray([
    Universe(1), Interval(-1.0, 1.0), Universe(4),
    BallInf(zeros(3), 1.0), Universe(9)]))

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
    k = 7  # falsification can run for a shorter time horizon
else
    k = 20
end
T = k * period
T_warmup = 2 * period;  # shorter time horizon for warm-up run

algorithm_plant = TMJets(abstol=2e-2, orderT=3, orderQ=1);

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
        result = "falsified"
    else
        silent || println("  The property may be satisfied.")
        result = "not falsified"
    end

    return sol, result
end;

benchmark(T=T_warmup, silent=true)  # warm-up
res = @timed benchmark(T=T)  # benchmark
sol, result = res.value
@assert (result == "falsified") "falsification failed"
println("Total analysis time:")
print_timed(res)

println("Simulation:")
res = @timed simulate(prob; T=T, trajectories=falsification ? 1 : 10,
                      include_vertices=!falsification)
sim = res.value
print_timed(res);

function plot_helper(vars)
    fig = plot()
    plot!(fig, project(safe_states, vars); color=:lightgreen, lab="safe")
    plot!(fig, project(initial_state(prob), vars); c=:cornflowerblue, alpha=1, lab="X₀")
    plot!(fig, sol; vars=vars, color=:yellow, lw=0, alpha=1, lab="")
    lab_sim = falsification ? "simulation" : ""
    plot_simulation!(fig, sim; vars=vars, color=:black, lab=lab_sim)
    return fig
end;

vars = (2, 7)
fig = plot_helper(vars)
plot!(fig; xlab="s_y", ylab="ϕ", leg=:bottomleft)
if falsification
    xlims!(-0.01, 1.15)
    ylims!(0.5, 1.01)
else
    xlims!(-0.55, 0.55)
    ylims!(-1.05, 1.05)
end
# Plots.savefig(fig, "Airplane.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

end
nothing
