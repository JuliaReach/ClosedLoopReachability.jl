export plot_simulation!

# convenience function for plotting simulation results
# use `output_map` to plot a linear combination of the state variables
function plot_simulation!(fig, sim::EnsembleSimulationSolution; vars=nothing, output_map=nothing,
                          kwargs...)
    # The main problem is that plotting trajectories one by one changes the plot
    # limits. Hence we store the plot limits from an existing figure and restore
    # them after plotting all trajectories.

    # argument checking
    got_vars = !isnothing(vars)
    got_output_map = !isnothing(output_map)
    color = get(kwargs, :color, :red)
    label = get(kwargs, :lab, "")

    if !got_vars && !got_output_map
        throw(ArgumentError("either `vars` or `output_map` should be specified"))
    elseif got_vars
        if vars isa AbstractVector
            if length(vars) != 2
                error("unexpected length of `vars`: $(length(vars))")
            end
            vars = (vars[1], vars[2])
        end
        _plot_function = _plot_simulation_vars!
        opts = vars
    else
        _plot_function = _plot_simulation_output_map!
        opts = output_map
    end

    # obtain x and y limits
    xl = Plots.xlims(fig)
    yl = Plots.ylims(fig)

    _plot_function(fig, sim, opts; color=color, label=label)

    # restore x and y limits
    Plots.xlims!(fig, xl)
    Plots.ylims!(fig, yl)

    return fig
end

function _plot_simulation_vars!(fig, sim, vars; color, label)
    for simulation in trajectories(sim)
        for piece in simulation
            Plots.plot!(fig, piece; idxs=vars, color=color, lab=label)
            label = ""  # overwrite to have exactly one label
        end
    end
end

function _plot_simulation_output_map!(fig, sim, output_map; color, label)
    # plot the first point only for the legend entry
    piece1 = first(first(trajectories(sim)))
    Plots.plot!(fig, [piece1.t[1]], [output_map(piece1.u[1])]; color=color, lab=label)

    for simulation in trajectories(sim)
        for piece in simulation
            Plots.plot!(fig, piece.t, output_map.(piece.u); color=color, lab="")
        end
    end
    return fig
end
