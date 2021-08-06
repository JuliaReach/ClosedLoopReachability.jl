using NeuralNetworkAnalysis, MAT
using NeuralNetworkAnalysis: UniformAdditivePostprocessing, SingleEntryVector
using Plots

@taylorize function unicycle!(dx, x, p, t)
    x₁, x₂, x₃, x₄, w, u₁, u₂ = x

    dx[1] = x₄ * cos(x₃)
    dx[2] = x₄ * sin(x₃)
    dx[3] = u₂
    dx[4] = u₁ + w
    dx[5] = zero(x[5])
    dx[6] = zero(x[6])
    dx[7] = zero(x[7])
    return dx
end

controller = read_nnet_mat(@modelpath("Sherlock-Benchmark-10-Unicycle",
                                      "controllerB_nnv.mat");
                           act_key="act_fcns");

X₀ = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5, -1e-4],
                    high=[9.55, -4.45, 2.11, 1.51, 1e-4])
U₀ = ZeroSet(2)
vars_idx = Dict(:state_vars=>1:4, :input_vars=>[5], :control_vars=>6:7)
ivp = @ivp(x' = unicycle!(x), dim: 7, x(0) ∈ X₀ × U₀)

period = 0.2

control_postprocessing = UniformAdditivePostprocessing(-20.0)

prob = ControlledPlant(ivp, controller, vars_idx, period;
                       postprocessing=control_postprocessing)

T = 5.0

target_set = HPolyhedron([HalfSpace(SingleEntryVector(1, 7, 1.0), 0.6),
                          HalfSpace(SingleEntryVector(1, 7, -1.0), 0.6),
                          HalfSpace(SingleEntryVector(2, 7, 1.0), 0.2),
                          HalfSpace(SingleEntryVector(2, 7, -1.0), 0.2),
                          HalfSpace(SingleEntryVector(3, 7, 1.0), 0.06),
                          HalfSpace(SingleEntryVector(3, 7, -1.0), 0.06),
                          HalfSpace(SingleEntryVector(4, 7, 1.0), 0.3),
                          HalfSpace(SingleEntryVector(4, 7, -1.0), 0.3)])

predicate_R_all = R -> R ⊆ target_set
predicate_sol_suff = sol -> predicate_R_all(sol[end]);

alg = TMJets(abstol=1e-15, orderT=10, orderQ=1)
alg_nn = DeepZ()
splitter_full = BoxSplitter([3, 1, 8, 1])

splitter1 = BoxSplitter([2, 1, 2, 1])
splitter_half = IndexedSplitter(Dict(1 => splitter1))

function benchmark(; T=T, silent::Bool=false, splitter)
    silent || println("flowpipe construction")
    res_sol = @timed sol = solve(prob, T=T, alg_nn=alg_nn, alg=alg,
                                 splitter=splitter)
    sol = res_sol.value
    silent || print_timed(res_sol)

    silent || println("property checking")
    res_pred = @timed predicate_sol_suff(sol)
    silent || print_timed(res_pred)
    if res_pred.value
        silent || println("The property is satisfied.")
    else
        silent || println("The property may be violated.")
    end
    return sol
end;

res = @timed sol = benchmark(splitter=splitter_full)
sol1 = res.value
println("total analysis time")
print_timed(res);
res2 = @timed sol = benchmark(splitter=splitter_half)
sol2 = res2.value
println("total analysis time")
print_timed(res2);


function p()

vars1 = (1, 2)

t1 = 1.0  # vary this
fig = plot()

for F in sol2
    t1 ∉ tspan(F) && continue
    for R in F
        t1 ∉ tspan(R) && continue
        plot!(R, vars=vars1, color=:green, alpha=0.1, lab="")
    end
end

for F in sol1
    t1 ∉ tspan(F) && continue
    for R in F
        t1 ∉ tspan(R) && continue
        plot!(R, vars=vars1, color=:yellow, alpha=0.1, lab="")
    end
end

label = "reach set at t = $t1 small splitting"
for F in sol2
    t1 ∉ tspan(F) && continue
    for R in F
        t1 ∉ tspan(R) && continue
        plot!(fig, overapproximate(R, Zonotope, t1), vars=vars1, color=:red, alpha=0.1, lab=label)
        label = ""
    end
end

label = "reach set at t = $t1 full splitting"
for F in sol1
    t1 ∉ tspan(F) && continue
    for R in F
        t1 ∉ tspan(R) && continue
        plot!(fig, overapproximate(R, Zonotope, t1), vars=vars1, color=:orange, lab=label)
        label = ""
    end
end
plot!()

end

p()
