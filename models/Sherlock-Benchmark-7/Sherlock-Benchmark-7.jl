using ReachabilityAnalysis, Plots

@taylorize function benchmark7!(du, u, p, t)
    local two = 2.0+zero(u[1])
    du[1] = u[3]^3 - u[2] + u[4]
    du[2] = u[3]
    du[3] = two
    du[4] = zero(u[4])
    return du
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[0.35, 0.45, 0.25, -0.01], high=[0.45, 0.55, 0.35, 0.01])

prob = @ivp(x' = benchmark7!(x), dim: 4, x(0) ∈ X₀)

# solve it
sol = solve(prob, T=2.0);
