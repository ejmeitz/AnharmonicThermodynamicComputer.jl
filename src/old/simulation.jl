export SamplingMode, Simulation

abstract type SamplingMode end

struct ResetMode <: SamplingMode end
struct SerialMode <: SamplingMode end


# Just a data container
struct Simulation{
    FT <: AbstractFloat,
    SM <: SamplingMode,
    T <: AbstractVecOrMat{FT},
}
    X::T
    η_buffer::T
    F_buffer::T
    Δt::FT
    t_final::FT
    kBT::FT
    drift_scale::FT
    noise_scale::FT
    M::Int
end


function Simulation(
    tnn::AllToAllTNN{FT},
    I::AbstractVecOrMat{FT},
    Δt::FT,
    t_final::FT,
    kBT::FT,
    M::Int,
    ::Type{SM};
    μ::FT = one(FT)
) where {FT, SM}

    @assert size(I, 1) == size(external_coupling(tnn), 1) "Input feature count must match the number of rows in W"

    n = round(Int, t_final / Δt)
    @assert isapprox(n * Δt, t_final) "t_final must be a multiple of Δt"

    # State of network
    X = if I isa AbstractVector
        similar(I, n_neurons(tnn))
    else
        similar(I, n_neurons(tnn), size(I, 2))
    end
    X .= FT(0.0)
    # Random number buffer for time-stepping
    η_storage = similar(X)
    # Force buffer for time-stepping
    F_storage = similar(X)

    drift_scale = (μ * Δt)
    noise_scale = sqrt(2 * kBT * drift_scale)

    return Simulation{FT,SM,typeof(X)}(X, η_storage, F_storage, Δt, t_final, kBT, drift_scale, noise_scale, M)
end

Base.length(sim::Simulation) = round(Int, sim.t_final / sim.Δt)

zero_state_vector!(sim::Simulation{FT}) where FT = (sim.X .= zero(FT))
zero_force_buffer!(sim::Simulation{FT}) where FT = (sim.F_buffer .= zero(FT))
zero_η_buffer!(sim::Simulation{FT}) where FT = (sim.η_buffer .= zero(FT))