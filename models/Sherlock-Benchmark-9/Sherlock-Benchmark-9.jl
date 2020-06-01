using ReachabilityAnalysis

@taylorize function benchmark9!(dx, x, p, t)
    u = one(x[4])
    aux = 0.1 * sin(x[3])
    dx[1] = x[2]
    dx[2] = -x[1] + aux
    dx[3] = x[4]
    dx[4] = u
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[0.6, -0.7, -0.4, 0.5], high=[0.7, -0.6, -0.3, 0.6])

prob = @ivp(x' = benchmark9!(x), dim: 4, x(0) ∈ X₀)

# solve it
sol = solve(prob, T=20.0);
