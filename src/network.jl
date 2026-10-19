export AllToAllTNN

abstract type ThermodynamicNeuralNetwork{FT <: AbstractFloat} end

const TNN = ThermodynamicNeuralNetwork


struct AllToAllTNN{
    FT,
    N <: AbstractVector{<:ThermodynamicNeuron{FT}},
    V <: AbstractVector{FT},
    M <: AbstractMatrix{FT},
} <: TNN{FT}
    neurons::N              # (n_neurons,)
    biases::V               # (n_neurons,)
    Jᵢⱼ::M                  # Internal coupling: (n_neurons, n_neurons)
    Wᵢⱼ::M                  # External coupling: (n_inputs, n_first_layer_neurons)
    fᵢ::M                   # (n_outputs, n_last_layer_neurons)
    n_layers::Int
    W_scale::M
    J_scale::M
end


# Not intended to be user-facing
function AllToAllTNN(neurons::N, biases::V, Jᵢⱼ::M, Wᵢⱼ::M, fᵢ::V, n_layers, Ws::M, Js::M) where {N, V, M}
    FT = eltype(Wᵢⱼ)
    return AllToAllTNN{FT, N, V, M}(neurons, biases, Jᵢⱼ, Wᵢⱼ, fᵢ, n_layers, Ws, Js)
end

bias(tnn::AllToAllTNN) = tnn.biases
internal_coupling(tnn::AllToAllTNN) = tnn.Jᵢⱼ
external_coupling(tnn::AllToAllTNN) = tnn.Wᵢⱼ
force_buffer(tnn::AllToAllTNN) = tnn.force_buffer
neurons(tnn::AllToAllTNN) = tnn.neurons

Base.length(tnn::AllToAllTNN) = length(neurons(tnn))
n_neurons(tnn::AllToAllTNN) = length(tnn)
n_layers(tnn::AllToAllTNN) = tnn.n_layers
neurons_in_first_layer(tnn::AllToAllTNN) = size(external_coupling(tnn), 2)
neurons_in_last_layer(tnn::AllToAllTNN) = neurons_in_first_layer(tnn) #! not the most general but whatever

zero_force_buffer!(tnn::AllToAllTNN{FT}) where FT = (tnn.force_buffer .= zero(FT))

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

function mutation_scales(
    connectivity_matrix,
    W::AbstractMatrix{FT},
) where FT
    n_inputs, n_first = size(W)

    # number of connections for each neuron
    counts = vec(FT.(sum(connectivity_matrix; dims=2))) # (n_neurons, )


    first_counts = view(counts, 1:n_first)
    first_counts .+= FT(n_inputs) # Account for external input

    W_scale = collect(transpose(inv.(sqrt.(first_counts))))  # (1, n_first)

    C = FT(0.5) .* (counts .+ transpose(counts))

    # Avoid division by zero for disconnected/isolated pairs
    J_scale = connectivity_matrix ./ sqrt.(max.(C, one(FT)))

    return W_scale, J_scale
end

function AllToAllTNN(
    n_layers::Int,
    neurons_per_layer::Int,
    n_outputs::Int,
    Wᵢⱼ::AbstractMatrix{FT}; # external coupling
    J₂ = one(FT), J₃ = zero(FT), J₄ = one(FT),
    σ = FT(0.1)
) where {FT <: AbstractFloat}

    @assert n_layers ≥ 1 && neurons_per_layer ≥ 1 "Layer and neuron counts must be positive"
    @assert size(Wᵢⱼ, 2) == neurons_per_layer "External coupling matrix must have the same number of columns as neurons in first layer"

    J₂, J₃, J₄ = FT(J₂), FT(J₃), FT(J₄)
    σ = FT(σ)

    n_neurons = n_layers * neurons_per_layer

    # Initialize things in a way that is agnostic
    # to the array backend (i.e. Array or CuArray)
    neuron = QuarticNeuron(FT(J₂), FT(J₃), FT(J₄))
    neurons = similar(Wᵢⱼ, typeof(neuron), n_neurons)
    fill!(neurons, neuron)

    biases = similar(Wᵢⱼ, n_neurons)
    randn!(biases)
    biases .*= σ

    connectivity_matrix = similar(Wᵢⱼ, Bool, n_neurons, n_neurons)
    all_to_all_connectivity_matrix!(connectivity_matrix, n_layers, neurons_per_layer)

    Jᵢⱼ = similar(Wᵢⱼ, n_neurons, n_neurons)
    randn!(Jᵢⱼ)
    Jᵢⱼ = FT(σ / sqrt(FT(2))) .* (Jᵢⱼ .+ transpose(Jᵢⱼ))
    Jᵢⱼ .*= connectivity_matrix

    fᵢ = similar(Wᵢⱼ, n_outputs, neurons_per_layer)
    randn!(fᵢ)
    fᵢ .*= σ

    W_scale, J_scale = mutation_scales(connectivity_matrix, Wᵢⱼ)

    return AllToAllTNN(
        neurons,
        biases,
        Jᵢⱼ,
        Wᵢⱼ,
        fᵢ,
        n_layers,
        W_scale, 
        J_scale
    )
end

function Base.similar(tnn::AllToAllTNN)
    return ThermodynamicNeuralNetwork(
        neurons(tnn), # Shared, treated as read-only
        similar(bias(tnn)),
        similar(internal_coupling(tnn)),
        similar(external_coupling(tnn)),
        similar(tnn.fᵢ),
        connectivity_matrix(tnn),
        n_layers(tnn),
    )
end

function copy_parameters!(dest::AllToAllTNN, src::AllToAllTNN)
    copyto!(bias(dest), bias(src))
    copyto!(internal_coupling(dest), internal_coupling(src))
    copyto!(external_coupling(dest), external_coupling(src))
    copyto!(dest.fᵢ, src.fᵢ)
    return dest
end


#############################


# Eqn 5
function V_internal(tnn::AllToAllTNN{FT}, x::AbstractVector{FT}) where FT
    V_local = mapreduce(potential_energy, +, neurons(tnn), x, bias(tnn))
    V_coupling = FT(0.5) * dot(x, internal_coupling(tnn) * x)
    return V_local + V_coupling
end

function V_internal(tnn::AllToAllTNN{FT}, X::AbstractVecOrMat{FT}) where FT
    V_local = sum(
        potential_energy.(neurons(tnn), X, bias(tnn));
        dims=1,
    )

    V_coupling = FT(0.5) .* sum(
        X .* (internal_coupling(tnn) * X);
        dims=1,
    )

    return V_local .+ V_coupling
end

function F_internal!(F::T, tnn::AllToAllTNN{FT}, X::T) where {FT, T <: AbstractVecOrMat{FT}}
    F .+= force.(neurons(tnn), X, bias(tnn)) # F_local
    F .-= internal_coupling(tnn) * X # F_coupling
    return F
end

# Eqn 6
function V_external(tnn::AllToAllTNN{FT}, X::T, I::T) where {FT, T <: AbstractVecOrMat{FT}}
    X_first = view(X, 1:neurons_in_first_layer(tnn), :)
    return sum(I .* (external_coupling(tnn) * X_first); dims=1)
end

function F_external!(F::T, tnn::AllToAllTNN{FT}, I::T) where {FT, T <: AbstractVecOrMat{FT}}
    F_first = view(F, 1:neurons_in_first_layer(tnn), :)
    F_first .-= transpose(external_coupling(tnn)) * I
    return F
end

potential_energy(tnn::AllToAllTNN{FT}, x::T, I::T) where {FT, T <: AbstractVecOrMat{FT}} = V_internal(tnn, x) .+ V_external(tnn, x, I)

function force!(F::T, tnn::AllToAllTNN{FT}, x::T, I::T) where {FT, T <: AbstractVecOrMat{FT}}
    F_internal!(F, tnn, x)
    F_external!(F, tnn, I)
    return F
end
