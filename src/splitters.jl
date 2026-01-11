abstract type AbstractSplitter end

# ==================
# standard splitter
# ==================

struct Splitter{S} <: AbstractSplitter
    split_fun::S
end

apply(s::Splitter, X) = s.split_fun(X)
Base.haskey(::Splitter, k::Int) = k == 1
Base.getindex(s::Splitter, k::Int) = k == 1 ? s : error("key $k not found")

function NoSplitter()
    split_fun = X0 -> [X0]
    return Splitter(split_fun)
end

function BoxSplitter(partition=nothing)
    if isnothing(partition)
        # default: one split per dimension
        split_fun = X -> split(box_approximation(X), 2 * ones(Int, dim(X)))
    else
        split_fun = X -> split(box_approximation(X), partition)
    end
    return Splitter(split_fun)
end

function ZonotopeSplitter(generators=nothing, splits=nothing)
    if isnothing(generators) && isnothing(splits)
        # default: one split per generator
        function split_fun(Z)
            p = ngens(Z)
            generators = 1:p
            splits = ones(Int, p)
            return split(Z, generators, splits)
        end
    else
        split_fun = Z -> split(Z, generators, splits)
    end
    return Splitter(split_fun)
end

# ==================================
# splitter for different iterations
# ==================================

struct IndexedSplitter <: AbstractSplitter
    index2splitter::Dict{Int,Splitter}
end

Base.haskey(s::IndexedSplitter, k::Int) = haskey(s.index2splitter, k)
Base.getindex(s::IndexedSplitter, k::Int) = getindex(s.index2splitter, k)

# ==============================
# splitter based on state space
# ==============================

struct SignSplitter <: AbstractSplitter
end

function apply(::SignSplitter, X::Interval{N}) where {N}
    l = low(X, 1)
    h = high(X, 1)
    if l < zero(N) && h > zero(N)
        return [LazySets.Interval(l, zero(N)), LazySets.Interval(zero(N), h)]
    else
        return [X]
    end
end
