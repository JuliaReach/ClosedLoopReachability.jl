# Plotting of the other dimensions:

vars=(1, 2)
fig = plot(xlab="θ₁", ylab="θ₂")
xlims!(-0.5, 2.0)
fig = plot_helper!(fig, vars, sol_lr, sim_lr, prob_lr, spec_lr, "less-robust")

vars=(1, 2)
fig = plot(xlab="θ₁", ylab="θ₂")
fig = plot_helper!(fig, vars, sol_mr, sim_mr, prob_mr, spec_mr, "more-robust")
