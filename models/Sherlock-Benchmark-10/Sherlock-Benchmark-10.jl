using ReachabilityAnalysis

@taylorize function benchmark10!(dx, x, p, t)
    u1, u2 = one(x[5]), one(x[5])
    dx[1] = x[4] * cos(x[3])
    dx[2] = x[4] * sin(x[3])
    dx[3] = u2
    dx[4] = u1 + x[5]
    dx[5] = zero(x[5]) # w
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[9.5, -4.5, 2.1, 1.5, -1e-4], high=[9.55, -4.45, 2.11, 1.55, 1e-1])

prob = @ivp(x' = benchmark10!(x), dim: 5, x(0) ∈ X₀)

# solve it
sol = solve(prob, T=10.0);
