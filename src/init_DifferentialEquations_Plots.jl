export plot_simulation!

# convenience function for plotting simulation results
# use `output_map` to plot a linear combination of the state variables
function plot_simulation!(fig, sim::EnsembleSimulationSolution; vars=nothing, output_map=nothing, kwargs...)
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
        _plot_function = _plot_simulation_vars!
        opts = vars
    else
        _plot_function = _plot_simulation_output_map!
        opts = output_map
    end

    # obtain x and y limits
    xl = Plots.xlims(fig)
    yl = Plots.ylims(fig)

    _plot_function(fig, sim, opts, color=color, label=label)

    # restore x and y limits
    Plots.xlims!(fig, xl)
    Plots.ylims!(fig, yl)

    return fig
end

function _plot_simulation_vars!(fig, sim, vars; color, label)
    for simulation in trajectories(sim)
        for piece in simulation
            Plots.plot!(fig, piece, vars=vars, color=color, lab=label)
            label = ""  # overwrite to have exactly one label
        end
    end
end

function _plot_simulation_output_map!(fig, sim, output_map::Vector{<:Real}; color, label)

    # dimension check
    numvars = length(output_map)
    traj1 = sim.solutions[1].trajectory[1]
    n = length(traj1.u[1])
    if numvars == n
        c0 = 0.0
        coeffs = output_map
    elseif numvars == n + 1
        c0 = output_map[1]
        coeffs = output_map[2:end]
    else
        throw(ArgumentError("the length of the `output_map` should be $n or $(n+1), got $numvars"))
    end

    f(t, P) = t -> c0 + sum(coeffs[i] * P(t)[i] for i in 1:n)
    isfirst = true

    for simulation in trajectories(sim)
        for piece in simulation
            dt = piece.t
            trange = range(dt[1], dt[end], length=length(dt))
            if isfirst
                # plot first point only for the legend entry
                trange1 = trange[1]:trange[1]
                Plots.plot!(fig, trange1, f.(trange1, Ref(piece)), color=color, lab=label)
                label = ""  # overwrite to have exactly one label
                isfirst = false
            end
            Plots.plot!(fig, trange, f.(trange, Ref(piece)), color=color, lab="")
        end
    end
    return fig
end
