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

# =====================
# Control preprocessing
# =====================

abstract type ControlPreprocessing end

struct NoPreprocessing <: ControlPreprocessing
end

apply(normalization::NoPreprocessing, x) = x

struct FunctionPreprocessing{F<:Function} <: ControlPreprocessing
    f::F

    function FunctionPreprocessing(f::F) where {F<:Function}
        return new{F}(f)
    end
end

function apply(funct::FunctionPreprocessing, x)
    return funct.f(x)
end

# ================
# Control problems
# ================


abstract type AbstractControlProblem end

"""
    ControlledPlant{ST, CT, XT, DT, PT} <: AbstractControlProblem

Struct representing a closed-loop controlled system.

### Fields

- `ivp`           -- initial-value problem
- `controller`    -- controller
- `vars`          -- dictionary storing state variables, input variables and control variables
- `period`        -- control period
- `normalization` -- normalization of the controller output
- `preprocessing` -- preprocessing of the controller input

### Parameters

- `ST`:  type of system
- `CT`:  type of controller
- `XT`:  type of initial condition
- `DT`:  type of variables
- `PT`:  type of period
- `CNT`: type of control normalization
- `CPT`: type of control preprocessing

### Notes

While typically the `controller` is a neural network, this struct does not
prescribe the type.
"""
struct ControlledPlant{ST, CT, XT, DT, PT, CNT, CPT} <: AbstractControlProblem
    ivp::InitialValueProblem{ST, XT}
    controller::CT
    vars::Dict{Symbol, DT}
    period::PT
    normalization::CNT
    preprocessing::CPT

    function ControlledPlant(ivp::InitialValueProblem{ST, XT},
                             controller::CT,
                             vars::Dict{Symbol, DT},
                             period::PT;
                             normalization::CNT=NoNormalization(),
                             preprocessing::CPT=NoPreprocessing()) where {ST, CT, XT, DT, PT, CNT, CPT}
        return new{ST, CT, XT, DT, PT, CNT, CPT}(ivp, controller, vars, period, normalization, preprocessing)
    end
end

plant(cp::ControlledPlant) = cp.ivp
MathematicalSystems.initial_state(cp::ControlledPlant) = initial_state(cp.ivp)
system(cp::ControlledPlant) = cp.ivp.s
controller(cp::ControlledPlant) = cp.controller
period(cp::ControlledPlant) = cp.period
control_normalization(cp::ControlledPlant) = cp.normalization
control_preprocessing(cp::ControlledPlant) = cp.preprocessing

state_vars(cp::ControlledPlant) = get(cp.vars, :state_vars, Int[])
input_vars(cp::ControlledPlant) = get(cp.vars, :input_vars, Int[])
control_vars(cp::ControlledPlant) = get(cp.vars, :control_vars, Int[])

# =============
# Specification
# =============

@with_kw struct Specification{NT, PT, ET}
    T::NT
    predicate::PT
    ext::ET = nothing
end

predicate(spec::Specification) = spec.predicate
time_horizon(spec::Specification) = spec.T
ext(spec::Specification) = spec.ext
