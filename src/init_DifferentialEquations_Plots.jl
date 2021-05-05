export plot_simulation!

# convenience function for plotting simulation results
function plot_simulation!(fig, sim::EnsembleSimulationSolution; vars, kwargs...)
    # The main problem is that plotting trajectories one by one changes the plot
    # limits. Hence we store the plot limits from an existing figure and restore
    # them after plotting all trajectories.

    # obtain x and y limits
    xl = Plots.xlims(fig)
    yl = Plots.ylims(fig)

    for simulation in trajectories(sim)
        for piece in simulation
            Plots.plot!(fig, piece, vars=vars, color=:red, lab="")
        end
    end

    # restore x and y limits
    Plots.xlims!(fig, xl)
    Plots.ylims!(fig, yl)

    return fig
end
