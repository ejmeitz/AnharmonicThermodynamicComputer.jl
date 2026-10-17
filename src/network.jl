export AllToAllTNN

struct ThermodynamicNeuralNetwork{
    FT <: AbstractFloat,
    N <: AbstractVector{<:ThermodynamicNeuron{FT}},
    V <: AbstractVector{FT},
    M <: AbstractMatrix{FT},
    C <: AbstractMatrix{Bool},
}
    neurons::N              # (n_neurons,)
    biases::V               # (n_neurons,)
    Jᵢⱼ::M                  # Internal coupling: (n_neurons, n_neurons)
    Wᵢⱼ::M                  # External coupling: (n_inputs, n_first_layer_neurons)
    fᵢ::V                   # (n_last_layer_neurons, )
    connectivity_matrix::C  # (n_neurons, n_neurons)
    force_buffer::V        # (n_neurons,)
    n_layers::Int
end

const TNN = ThermodynamicNeuralNetwork

# Not intended to be user-facing
function ThermodynamicNeuralNetwork(neurons::N, biases::V, Jᵢⱼ::M, Wᵢⱼ::M, fᵢ::V, cm::C, fb::V, n_layers) where {N, V, M, C}
    FT = eltype(Wᵢⱼ)
    return TNN{FT, N, V, M, C}(neurons, biases, Jᵢⱼ, Wᵢⱼ, fᵢ, cm, fb, n_layers)
end

bias(tnn::TNN) = tnn.biases
internal_coupling(tnn::TNN) = tnn.Jᵢⱼ
external_coupling(tnn::TNN) = tnn.Wᵢⱼ
force_buffer(tnn::TNN) = tnn.force_buffer
neurons(tnn::TNN) = tnn.neurons

Base.length(tnn::TNN) = length(neurons(tnn))
n_neurons(tnn::TNN) = length(tnn)
n_layers(tnn::TNN) = tnn.n_layers
neurons_in_first_layer(tnn::TNN) = size(external_coupling(tnn), 2)
neurons_in_last_layer(tnn::TNN) = neurons_in_first_layer(tnn) #! not the most general but whatever

zero_force_buffer!(tnn::TNN{FT}) where FT = (tnn.force_buffer .= zero(FT))

function all_to_all_connectivity_matrix!(cm, n_layers::Int, neurons_per_layer::Int)

    cm .= false

    for layer_idx in 1:(n_layers - 1)
        # Index of the first neuron in this layer
        first_node_idx = (layer_idx - 1) * neurons_per_layer + 1
        # Indices of neurons in this layer
        current_nodes = first_node_idx:(first_node_idx + neurons_per_layer - 1)
        # Indices of neurons in the next layer
        next_nodes = current_nodes .+ neurons_per_layer

        cm[current_nodes, next_nodes] .= true
        cm[next_nodes, current_nodes] .= true
    end

    return cm
end

function AllToAllTNN(
    n_layers::Int,
    neurons_per_layer::Int,
    Wᵢⱼ::AbstractMatrix{FT}; # external coupling
    J₂ = one(FT), J₃ = zero(FT), J₄ = one(FT)
) where {FT <: AbstractFloat}

    @assert n_layers ≥ 1 && neurons_per_layer ≥ 1 "Layer and neuron counts must be positive"
    @assert size(Wᵢⱼ, 2) == neurons_per_layer "External coupling matrix must have the same number of columns as neurons in first layer"

    J₂, J₃, J₄ = FT(J₂), FT(J₃), FT(J₄)

    n_neurons = n_layers * neurons_per_layer

    # Initialize things in a way that is agnostic
    # to the array backend (i.e. Array or CuArray)
    neuron = QuarticNeuron(FT(J₂), FT(J₃), FT(J₄))
    neurons = similar(Wᵢⱼ, typeof(neuron), n_neurons)
    fill!(neurons, neuron)

    biases = similar(Wᵢⱼ, n_neurons)
    randn!(biases) #! initialize someother way?

    force_buffer = similar(biases)
    force_buffer .= FT(0.0)

    connectivity_matrix = similar(Wᵢⱼ, Bool, n_neurons, n_neurons)
    all_to_all_connectivity_matrix!(connectivity_matrix, n_layers, neurons_per_layer)

    Jᵢⱼ = similar(Wᵢⱼ, n_neurons, n_neurons)
    randn!(Jᵢⱼ) #! initialize someother way?
    Jᵢⱼ = FT(0.5) * (Jᵢⱼ + transpose(Jᵢⱼ))
    Jᵢⱼ .*= connectivity_matrix

    fᵢ = similar(Wᵢⱼ, neurons_per_layer)
    randn!(fᵢ) #! initialize someother way?

    return ThermodynamicNeuralNetwork(
        neurons,
        biases,
        Jᵢⱼ,
        Wᵢⱼ,
        fᵢ,
        connectivity_matrix,
        force_buffer,
        n_layers
    )
end


# Eqn 5
function V_internal(tnn::TNN{FT}, x::AbstractVector{FT}) where FT
    V_local = mapreduce(potential_energy, +, neurons(tnn), x, bias(tnn))
    V_coupling = FT(0.5) * dot(x, internal_coupling(tnn) * x)
    return V_local + V_coupling
end

function F_internal!(tnn::TNN{FT}, x::AbstractVector{FT}) where FT

    F_buffer = force_buffer(tnn)

    # F_local
    F_buffer .+= force.(neurons(tnn), x, bias(tnn))
    # F_coupling
    F_buffer .-= internal_coupling(tnn) * x

    return tnn
end

# Eqn 6
function V_external(tnn::TNN{FT}, x::AbstractVector{FT}, I::AbstractVector{FT}) where FT
    Wᵢⱼ = external_coupling(tnn)
    @assert length(I) == size(Wᵢⱼ, 1) "Input vector length must match the number of rows in the external coupling matrix"
    x_first = view(x, 1:neurons_in_first_layer(tnn))
    return dot(I, Wᵢⱼ * x_first)
end

function F_external!(tnn::TNN{FT}, I::AbstractVector{FT}) where FT
    Wᵢⱼ = external_coupling(tnn)
    F_first = view(force_buffer(tnn), 1:neurons_in_first_layer(tnn))
    F_first .-= transpose(Wᵢⱼ) * I
    return tnn
end

V(tnn::TNN{FT}, x::AbstractVector{FT}, I::AbstractVector{FT}) where FT = V_internal(tnn, x) + V_external(tnn, x, I)

function F!(tnn::TNN{FT}, x::AbstractVector{FT}, I::AbstractVector{FT}) where FT
    zero_force_buffer!(tnn)
    F_internal!(tnn, x)
    F_external!(tnn, I)
    return tnn
end
