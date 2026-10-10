module VerticalCAS

using ClosedLoopReachability, LinearAlgebra
import Plots, DisplayAs
using ReachabilityBase.CurrentPath: @current_path
using ReachabilityBase.Timing: print_timed
using ClosedLoopReachability: FeedforwardNetwork
using Plots: plot, plot!

struct State{T}
    state::T  # state for (h, hdot0) variables
    τ::Float64
    adv::Symbol
end;

const index2advisory = [:COC, :DNC, :DND, :DES1500, :CL1500,
                        :SDES1500, :SCL1500, :SDES2500, :SCL2500];

const advisory2set = Dict{Symbol,LazySet}()
advisory2set[:COC] = EmptySet(1)
advisory2set[:DNC] = HalfSpace([1.0], 0.0)
advisory2set[:DND] = HalfSpace([-1.0], 0.0)
advisory2set[:DES1500] = HalfSpace([1.0], -1500.0)
advisory2set[:CL1500] = HalfSpace([-1.0], -1500.0)
advisory2set[:SDES1500] = HalfSpace([1.0], -1500.0)
advisory2set[:SCL1500] = HalfSpace([-1.0], -1500.0)
advisory2set[:SDES2500] = HalfSpace([1.0], -2500.0)
advisory2set[:SCL2500] = HalfSpace([-1.0], -2500.0);

const advisory2controller = Dict{Symbol,FeedforwardNetwork}()

path_prefix = @current_path("VerticalCAS", "")
for i in 1:9
    path = joinpath(path_prefix, "VerticalCAS_controller_$(i).polar")
    adv = index2advisory[i]
    advisory2controller[adv] = read_POLAR(path)
end;

const normalization_additive = -[0.0, 0, 20]
const normalization_multiplicative = 1.0 ./ [16000.0, 5000, 40]
const normalization_multiplicative_X = diagm(normalization_multiplicative)

function normalize(x::AbstractVector)
    y = x .+ normalization_additive
    z = normalization_multiplicative .* y
    return z
end

function normalize(X::LazySet)
    Y = translate(X, normalization_additive)
    Z = linear_map(normalization_multiplicative_X, Y)
    return Z
end;

function next_adv(X::LazySet, τ, adv; algorithm_controller=DeepZ())
    Y = cartesian_product(X, Singleton([τ]))
    Y = normalize(Y)
    out = forward(Y, advisory2controller[adv], algorithm_controller)
    imax = argmax(high(out))
    return index2advisory[imax]
end

function next_adv(X::Singleton, τ, adv; algorithm_controller=nothing)
    v = vcat(element(X), τ)
    v = normalize(v)
    u = forward(v, advisory2controller[adv])
    imax = argmax(u)
    return index2advisory[imax]
end;

const g = 32.2
const acc_central = Dict(:COC => 0.0, :DNC => -7g / 24, :DND => 7g / 24,
                         :DES1500 => -7g / 24, :CL1500 => 7g / 24, :SDES1500 => -g / 3,
                         :SCL1500 => g / 3, :SDES2500 => -g / 3, :SCL2500 => g / 3);

function next_acc(X::State, adv; acc=acc_central)
    # Project on hdot and transform units from ft/s to ft/min:
    hdot = 60 * _interval(X.state, 2)

    # New advisory:
    adv′ = X.adv

    # Check whether the current state complies with the advisory:
    comply = hdot ⊆ advisory2set[adv′]

    return (comply && adv == adv′) ? 0.0 : acc[adv′]
end;

const Δτ = 1.0
const A = [1 -Δτ; 0 1]  # dynamics matrix (h, \dot{h}_0)

function VerticalCAS!(out::Vector{<:State}, kmax::Int; acc, algorithm_controller)
    # Unpack the initial state:
    X0 = first(out)
    S = X0.state
    τ = X0.τ
    adv = X0.adv

    for k in 1:kmax
        # Get the next advisory and acceleration:
        adv′ = next_adv(S, τ, adv; algorithm_controller=algorithm_controller)
        X = State(S, τ, adv′)
        hddot = next_acc(X, adv; acc=acc)

        # Compute and store the next state:
        b = [-hddot * Δτ^2 / 2, hddot * Δτ]
        S′ = affine_map(A, S, b)
        τ′ = τ - Δτ
        X′ = State(S′, τ′, adv′)
        push!(out, X′)

        # Update the current state:
        S = S′
        τ = τ′
        adv = adv′
    end
    return out
end;

const h_0 = Interval(-133.0, -129.0)
const hdot0_0 = [-19.5, -22.5, -25.5, -28.5]
const τ_0 = 25.0
const adv_0 = :COC;

unsafe_states = HalfSpace([0.0, 1.0], 100.0) ∩ HalfSpace([0.0, -1.0], 100.0)

predicate_set(R) = isdisjoint(R, unsafe_states)

predicate(sol) = all(predicate_set(R) for R in sol)

kmax = 10
kmax_warmup = 2;  # shorter time horizon for warm-up run

function get_initial_states(hdot0_0i)
    S0 = convert(Zonotope, cartesian_product(h_0, Singleton([hdot0_0i])))
    return State(S0, τ_0, adv_0)
end;

function simulate_VerticalCAS(X0::State; kmax)
    out = [X0]
    sizehint!(out, kmax + 1)
    VerticalCAS!(out, kmax; acc=acc_central, algorithm_controller=DeepZ())
    return out
end;

_interval(X::LazySet, i) = Interval(extrema(X, i)...);

function _project(X::Vector{State{T}}) where {T<:Singleton}
    return [Singleton([Xi.τ, Xi.state.element[1]]) for Xi in X]
end

function _project(X::Vector{State{T}}) where {T<:LazySet}
    return [Singleton([Xi.τ]) × _interval(Xi.state, 1) for Xi in X]
end;

function benchmark(X0; kmax, silent::Bool=false)
    res = @timed begin
        seq = simulate_VerticalCAS(X0; kmax=kmax)
        _project(seq)
    end
    sol = res.value
    silent || print_timed(res)

    silent || println("Property checking:")
    res = @timed predicate(sol)
    silent || print_timed(res)
    if res.value
        silent || println("  The property is satisfied.")
        result = "verified"
    else
        silent || println("  The property is violated.")
        result = "falsified"
    end

    return sol, result
end;

println("Running flowpipe construction with central advisories:")
sol_all = []
for hdot0_0i in hdot0_0
    println("Running instance hdot0(0) = $hdot0_0i:")
    X0 = get_initial_states(hdot0_0i)
    benchmark(X0; kmax=kmax_warmup, silent=true)  # warm-up
    res = @timed benchmark(X0; kmax=kmax)  # benchmark
    sol, result = res.value
    push!(sol_all, sol)
    if hdot0_0i ∈ [-19.5, -22.5]
        @assert (result == "verified") "verification failed"
    elseif hdot0_0i ∈ [-25.5, -28.5]
        @assert (result == "falsified") "falsification failed"
    end
    println("Total analysis time:")
    print_timed(res)
end

function extend_x(X::Singleton; Δ=Δτ)
    return LineSegment(element(X) .- [Δ, 0], element(X))
end

function extend_x(cp::CartesianProduct; Δ=Δτ)
    x = first(element(first(cp)))
    X = Interval(x - Δ, x)
    return CartesianProduct(X, LazySets.second(cp))
end

function extend_x(sol_all::Vector)
    return [vcat([extend_x(X) for X in F[1:(end - 1)]], extend_x(F[end]; Δ=0.1)) for F in sol_all]
end

sol_all = extend_x(sol_all);

function plot_helper()
    fig = plot(ylab="h (vertical distance)", xlab="τ (time to reach horizontally)",
               xflip=true, leg=:topright, xticks=14:25)
    unsafe_states_projected = cartesian_product(Universe(1),
                                                project(unsafe_states, [2]))
    plot!(fig, unsafe_states_projected; alpha=0.2, c=:red, lab="unsafe")
    return fig
end;

fig = plot_helper()
for (i, c) in [(1, :brown), (2, :green), (3, :orange), (4, :cyan)]
    lab = "h_0′ = $(hdot0_0[i])"
    for o in sol_all[i]
        plot!(fig, o; lw=2, alpha=1, seriestype=:shape, c=c, lab=lab)
        lab = ""
    end
end
plot!(fig, xlims=(14.9, 25), ylims=(-310, -70))
# Plots.savefig("VerticalCAS.png")  # command to save the plot to a file
fig = DisplayAs.Text(DisplayAs.PNG(fig))

end
nothing
