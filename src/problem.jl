# =====================
# Control normalization
# =====================

abstract type ControlNormalization end

struct NoNormalization <: ControlNormalization
end

apply(normalization::NoNormalization, x) = x

struct UniformAdditiveNormalization{N<:Number} <: ControlNormalization
    shift::N

    function UniformAdditiveNormalization(shift::N) where {N<:Number}
        if shift == zero(N)
            @warn("zero additive control normalization is discouraged; " *
                  "use `NoNormalization` instead")
        end
        return new{N}(shift)
    end
end

function apply(normalization::UniformAdditiveNormalization, x)
    return x .+ normalization.shift
end

# ================
# Control problems
# ================


abstract type AbstractNeuralNetworkControlProblem end

"""
    ControlledPlant{ST, XT, DT, PT} <: AbstractNeuralNetworkControlProblem

Struct representing a closed-loop neural-network controlled system.

### Fields

- `ivp`           -- initial-value problem
- `controller`    -- neural-network controller
- `vars`          -- dictionary storing state variables, input variables and control variables
- `period`        -- control period
- `normalization` -- normalization of the controller output

### Parameters

- `ST`:  type of system
- `XT`:  type of initial condition
- `DT`:  type of variables
- `PT`:  type of period
- `CNT`: type of control normalization
"""
struct ControlledPlant{ST, XT, DT, PT, CNT} <: AbstractNeuralNetworkControlProblem
    ivp::InitialValueProblem{ST, XT}
    controller::Network
    vars::Dict{Symbol, DT}
    period::PT
    normalization::CNT

    function ControlledPlant(ivp::InitialValueProblem{ST, XT},
                             controller::Network,
                             vars::Dict{Symbol, DT},
                             period::PT,
                             normalization::CNT=NoNormalization()) where {ST, XT, DT, PT, CNT}
        return new{ST, XT, DT, PT, CNT}(ivp, controller, vars, period, normalization)
    end
end

plant(cp::ControlledPlant) = cp.ivp
system(cp::ControlledPlant) = cp.ivp.s
controller(cp::ControlledPlant) = cp.controller
period(cp::ControlledPlant) = cp.period
control_normalization(cp::ControlledPlant) = cp.normalization

state_vars(cp::ControlledPlant) = get(cp.vars, :state_vars, Int[])
input_vars(cp::ControlledPlant) = get(cp.vars, :input_vars, Int[])
control_vars(cp::ControlledPlant) = get(cp.vars, :control_vars, Int[])
