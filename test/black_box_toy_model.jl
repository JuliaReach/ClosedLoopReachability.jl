using ClosedLoopReachability

@taylorize function sys!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = zero(x[2])
    return dx
end;

function oscillator(X::LazySet)
    if high(X)[1] > 1
        if low(X)[1] < 1
            res = Interval(-0.1, 0.1)
        else
            res = Interval(-0.1, -0.1)
        end
    else
        res = Interval(0.1, 0.1)
    end
    return res
end;

function oscillator(x::Vector)
    if x[1] > 1
        res = [-0.1]
    else
        res = [0.1]
    end
    return res
end;
controller = BlackBoxController(oscillator);

X₀ = Interval(0.95, 0.99);
U = ZeroSet(1);
vars_idx = Dict(:state_vars=>[1], :control_vars=>2);
ivp = @ivp(x' = sys!(x), dim: 2, x(0) ∈ X₀ × U);
period = 0.1;
prob = ControlledPlant(ivp, controller, vars_idx, period);
alg = TMJets21b(abstol=1e-10, orderT=8, orderQ=2, adaptive=true);
alg_nn = BlackBoxSolver();
N = 10;
T = N*period;

sol_raw = solve(prob, T=T, alg_nn=alg_nn, alg=alg);

# import DifferentialEquations
# sim = simulate(prob, T=T; trajectories=10, include_vertices=true);

# using Plots
# fig = plot()
# vars = (0, 1)
# plot!(fig, sol_raw, vars=vars, c=:yellow, lw=2.0, ls=:dash)
# plot_simulation!(fig, sim; vars=vars, color=:red, lab="");
# savefig("oscillator_$(N)_steps")
