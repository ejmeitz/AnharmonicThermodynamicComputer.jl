export SamplingMode, Simulation

abstract type SamplingMode end

struct ResetMode <: SamplingMode end
struct SerialMode <: SamplingMode end


# Just a data container
struct Simulation{
    FT <: AbstractFloat,
    T <: AbstractVecOrMat{FT},
    SM <: SamplingMode
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
    mode::Type{SM}
end

function Simulation(
    tnn::TNN{FT},
    I::AbstractVecOrMat{FT},
    Δt::FT,
    t_final::FT,
    kBT::FT,
    M::Int,
    ::Type{SM};
    μ::FT = one(FT)
) where {FT, SM}

    @assert size(I, 1) == size(external_coupling(tnn), 1) "Input feature count must match the number of rows in W"

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

    return Simulation(X, η_storage, F_storage, Δt, t_final, kBT, drift_scale, noise_scale, M, SM)
end

Base.length(sim::Simulation) = round(Int, sim.t_final / sim.Δt)

zero_state_vector!(sim::Simulation{FT}) where FT = (sim.X .= zero(FT))
zero_force_buffer!(sim::Simulation{FT}) where FT = (sim.F_buffer .= zero(FT))
zero_η_buffer!(sim::Simulation{FT}) where FT = (sim.η_buffer .= zero(FT))


#! look into Evolutionary.jl
"""

If the input, `z`, is:
- Scalar: M

"""
function train_genetic(
    n_candidates::Int,
    n_layers::Int,
    neurons_per_layer::Int,
    Wᵢⱼ::AbstractMatrix{FT}, # (n_inputs x n_neurons_first_layer)
    z::AbstractVecOrMat{FT}, # (n_inputs x n_points) or (n_inputs, )
    M::Int,
    ::ResetMode,
    target_function
)

    @assert length(z) == size(Wᵢⱼ, 1) "Input vector `z` must have same length as number of rows in external coupling matrix Wᵢⱼ"

    candidates = [AllToAllTNN(n_layers, neurons_per_layer, Wᵢⱼ) for _ in 1:n_candidates]

    ground_truth = target_function.(z_k)

    for candidate in candidates
        y_pred = _estimate!(candidate, Δt, t_final, z, M, ResetMode)
        y_actual = t
        loss = 
    end

end