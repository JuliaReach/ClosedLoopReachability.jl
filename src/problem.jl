abstract type AbstractNeuralNetworkControlProblem end

"""
    ControlledPlant{ST, XT, DT, PT} <: AbstractNeuralNetworkControlProblem

Struct representing a closed-loop neural-network controlled system.

### Fields

- `ivp`        -- initial-value problem
- `controller` -- neural-network controller
- `vars`       -- dictionary storing state variables, input variables and control variables
- `period`     -- control period

### Parameters

- `ST`: type of system
- `XT`: type of initial condition
- `DT`: type of variables
- `PT`: type of period
"""
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

state_vars(cp::ControlledPlant) = get(cp.vars, :state_vars, nothing)
input_vars(cp::ControlledPlant) = get(cp.vars, :input_vars, nothing)
control_vars(cp::ControlledPlant) = get(cp.vars, :control_vars, nothing)
