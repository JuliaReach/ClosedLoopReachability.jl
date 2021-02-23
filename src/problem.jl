abstract type AbstractNeuralNetworkControlProblem end

# ST: type of system
# XT: type of initial condition
struct ControlledPlant{ST, XT, IV<:InitialValueProblem{ST, XT}, DT} <: AbstractNeuralNetworkControlProblem
    ivp::IV
    controller::Network
    vars::Dict{Symbol, DT}
end

plant(cp::ControlledPlant) = cp.ivp
system(cp::ControlledPlant) = cp.ivp.s
controller(cp::ControlledPlant) = cp.controller

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
