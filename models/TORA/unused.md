# Further plots:

vars = (1, 3)
fig = plot(xlab="x₁", ylab="x₃")
plot_helper(fig, vars)

#-

vars = (1, 4)
fig = plot(xlab="x₁", ylab="x₄")
plot_helper(fig, vars)

#-

vars=(2, 3)
fig = plot(xlab="x₂", ylab="x₃")
plot_helper(fig, vars)

#-

vars=(2, 4)
fig = plot(xlab="x₂", ylab="x₄")
plot_helper(fig, vars)

#-

vars = (0, 1)
fig = plot(xlab="t", ylab="x₁")
plot_helper(fig, vars)

#-

vars=(0, 2)
fig = plot(xlab="t", ylab="x₂")
plot_helper(fig, vars)

#-

vars=(0, 3)
fig = plot(xlab="t", ylab="x₃")
plot_helper(fig, vars)

#-

vars=(0, 4)
fig = plot(xlab="t", ylab="x₄")
plot_helper(fig, vars)

#-

# Plot the control signals for each simulation run:

tdom = range(0, 20, length=length(controls(sim, 1)))
fig = plot(xlab="t", ylab="u")
for i in 1:length(sim)
    plot!(fig, tdom, [c[1] for c in controls(sim, i)]; lab="")
end
fig = DisplayAs.Text(DisplayAs.PNG(fig))
