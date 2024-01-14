# Alternative matrix with only sin/cos but with postprocessing:
Rϕθ_ = (ϕ, θ) -> [cos(θ)  sin(θ) * sin(ϕ)   sin(θ) * cos(ϕ);
                       0  cos(θ) * cos(ϕ)  -cos(θ) * sin(ϕ);
                       0           sin(ϕ)            cos(ϕ)]

# Alternative matrix with postprocessing (to replace in the `Airplane!` function):
mat_2 = 1 / cos(θ) * Rϕθ_(ϕ, θ)

# Unused constants (terms are simplified instead):
const Ix = 1.0
const Iy = 1.0
const Iz = 1.0
const Ixz = 0.0

# Plot of the other two relevant dimensions

vars = (8, 9)
fig = plot(xlab="θ", ylab="ψ", leg=:bottom)
fig = plot_helper!(fig, vars)
if falsification
    xlims!(0.9, 1.01)
    ylims!(0.85, 1.01)
else
    xlims!(-1.05, 1.05)
    ylims!(-1.05, 1.05)
end
fig = DisplayAs.Text(DisplayAs.PNG(fig))
savefig("Airplane-x8-x9.png")  # command to save the plot to a file
