using ReachabilityAnalysis

@taylorize function ACC!(dx, x, p, t)
    local u, a_lead, a_ego = 0.0001, -2., -1.
    # x[1] = x_lead
    # x[2] = v_lead
    # x[3] = γ_lead
    # x[4] = x_ego
    # x[5] = v_ego
    # x[6] = γ_ego

    aux = 2 * u
    
    dx[1] = x[2]
    dx[2] = x[3]
    dx[3] = -2 * x[3] + 2 * a_lead - aux * x[2] * x[2]
    dx[4] = x[5]
    dx[5] = x[6]
    dx[6] = -2 * x[6] + 2 * a_ego - aux * x[5] * x[5]
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[90., 32, 0, 10, 30, 0], high=[110, 32.2, 0, 11, 30.2, 0])

prob = @ivp(x' = ACC!(x), dim: 6, x(0) ∈ X₀)

# solve it
sol = solve(prob, T=0.1);
