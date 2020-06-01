using ReachabilityAnalysis

@taylorize function ACC!(dx, x, p, t)
    local u, a_lead, a_ego = 0.0001, -2., -1.
    x_lead, v_lead, γ_lead, x_ego, v_ego, γ_ego = x
    
    dx[1] = v_lead
    dx[2] = γ_lead
    dx[3] = 2 * (a_lead - γ_lead) - u * v_lead^2
    dx[4] = v_ego
    dx[5] = γ_ego
    dx[6] = 2 * (a_ego - γ_ego) - u * v_ego^2
    return dx
end

# define the initial-value problem
X₀ = Hyperrectangle(low=[90., 32, 0, 10, 30, 0], high=[110, 32.2, 0, 11, 30.2, 0])

prob = @ivp(x' = ACC!(x), dim: 6, x(0) ∈ X₀)

# solve it
sol = solve(prob, T=0.1);
