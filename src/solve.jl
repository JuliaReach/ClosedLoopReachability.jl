using MathematicalSystems

using ReachabilityAnalysis: _check_dim, _get_tspan, _get_cpost, _default_cpost,
                            ReachSolution, InitialValueProblem
using NeuralVerification: Network

import ReachabilityAnalysis: solve

abstract type AbstractNeuralNetworkControlProblem end

# ST: type of system
# XT: type of initial condition
struct ControlledPlant{ST, XT, IV<:InitialValueProblem{ST, XT}, DT} <: AbstractNeuralNetworkControlProblem
    ivp::IV
    controller::Network
    vars::Dict{Symbol, DT}
end

plant(plant::ControlledPlant) = plant.ivp
controller(plant::ControlledPlant) = plant.controller


function state_vars(plant::ControlledPlant)
    try
        plant.vars[:state_vars]
    catch error
        isa(error, KeyError) && println("key `:state_vars` not found")
    end
end

function input_vars(plant::ControlledPlant)
    try
        plant.vars[:input_vars]
    catch error
        isa(error, KeyError) && println("key `:input_vars` not found")
    end
end

function control_vars(plant::ControlledPlant)
    try
        plant.vars[:control_vars]
    catch error
        isa(error, KeyError) && println("key `:control_vars` not found")
    end
end

#=
NOTES:

- Add default neural network solver.

=#
"""
    solve(prob::AbstractNeuralNetworkControlProblem, args...; kwargs...)

Solves the neural network controlled problem defined by `prob`

### Input

- `prob`   -- neural network controlled problem

Additional options are passed as arguments or keyword arguments; see the notes
below for details. See the online documentation for examples.

### Output

The solution of a reachability problem controlled by a neural network.

### Notes

- Use the `tspan` keyword argument to specify the time span; it can be:
    - a tuple,
    - an interval, or
    - a vector with two components.

- Use the `T` keyword argument to specify the time horizon; the initial time is
  then assumed to be zero.
  
- Use the `Tsample` or `sampling_time` keyword argument to specify the sampling
time for the model.

- Use the `alg_nn` keyword argument to specify the the neural network solver.

"""
function solve(prob::AbstractNeuralNetworkControlProblem, args...; kwargs...)
    ivp = plant(prob)

    # check that the dimension of the system and the initial condition match
    _check_dim(ivp)

    # get time span
    tspan = _get_tspan(args...; kwargs...)

    # get the continuous post or find a default one
    cpost = _get_cpost(ivp, args...; kwargs...)
    if cpost == nothing
        cpost = _default_cpost(ivp, tspan; kwargs...)
    end
    
    # extract the sampling time 
    Tsample = _get_Tsample(; kwargs...)
    
    solver = _get_alg_nn(args...; kwargs...)
    
    if haskey(kwargs, :apply_initial_control)
        init_ctrl = kwargs[:apply_initial_control]
    else
        init_ctrl = true
    end

    sol = _solve(prob, cpost, solver, tspan, Tsample, init_ctrl)
    
    d = Dict{Symbol, Any}(:solver=>solver)
    return ReachSolution(sol, cpost, d)
end

function _get_Tsample(; kwargs...)
    if haskey(kwargs, :sampling_time)
        Tsample = kwargs[:sampling_time]
    elseif haskey(kwargs, :Tsample)
        Tsample = kwargs[:Tsample]
    else
        throw(ArgumentError("the sampling time `Tsample` should be specified, but was not found"))
    end
    return Tsample
end

function _get_alg_nn(args...; kwargs...)
    if haskey(kwargs, :alg_nn)
        solver = kwargs[:alg_nn]
    else
        throw(ArgumentError("the solver for the neural network `alg_nn` should be specified, but was not found"))
    end
    return solver
end
