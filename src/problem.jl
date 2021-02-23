abstract type AbstractNeuralNetworkControlProblem end

# ST: type of system
# XT: type of initial condition
# DT: type of variables
# PT: type of period
struct ControlledPlant{ST, XT, DT, PT} <: AbstractNeuralNetworkControlProblem
    ivp::InitialValueProblem{ST, XT}
    controller::Network
    vars::Dict{Symbol, DT}
    period::PT
end

plant(cp::ControlledPlant) = cp.ivp
system(cp::ControlledPlant) = cp.ivp.s
controller(cp::ControlledPlant) = cp.controller
period(cp::ControlledPlant) = cp.period

function state_vars(cp::ControlledPlant)
    try
        cp.vars[:state_vars]
    catch error
        if isa(error, KeyError)
            println("key `:state_vars` not found")
        else
            println(error)
        end
    end
end

function input_vars(cp::ControlledPlant)
    try
        cp.vars[:input_vars]
    catch error
        if isa(error, KeyError)
            println("key `:input_vars` not found")
        else
            println(error)
        end
    end
end

function control_vars(cp::ControlledPlant)
    try
        cp.vars[:control_vars]
    catch error
        if isa(error, KeyError)
            println("key `:control_vars` not found")
        else
            println(error)
        end
    end
end
