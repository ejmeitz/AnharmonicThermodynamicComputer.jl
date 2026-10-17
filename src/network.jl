export AllToAllTNN

struct ThermodynamicNeuralNetwork{
    N <: AbstractVector{<:ThermodynamicNeuron},
    B <: AbstractVector{Float64},
    J <: AbstractMatrix{Float64},
    W <: AbstractMatrix{Float64},
    C <: AbstractMatrix{Bool},
}
    neurons::N              # (n_neurons,)
    biases::B               # (n_neurons,)
    Jᵢⱼ::J                  # Internal coupling: (n_neurons, n_neurons)
    Wᵢⱼ::W                  # External coupling: (n_inputs, n_first_layer_neurons)
    connectivity_matrix::C  # (n_neurons, n_neurons)
end

const TNN = ThermodynamicNeuralNetwork

neurons(tnn::TNN) = tnn.neurons
Base.length(tnn::TNN) = length(neurons(tnn))
n_neurons(tnn::TNN) = length(tnn)

bias(tnn::TNN) = tnn.biases
internal_coupling(tnn::TNN) = tnn.Jᵢⱼ
external_coupling(tnn::TNN) = tnn.Wᵢⱼ

function all_to_all_connectivity_matrix(n_layers::Int, neurons_per_layer::Int)
    n_neurons = n_layers * neurons_per_layer
    connectivity_matrix = fill(false, n_neurons, n_neurons)

    for layer_idx in 1:(n_layers - 1)
        # Index of the first neuron in this layer
        first_node_idx = (layer_idx - 1) * neurons_per_layer + 1
        # Indices of neurons in this layer
        current_nodes = first_node_idx:(first_node_idx + neurons_per_layer - 1)
        # Indices of neurons in the next layer
        next_nodes = current_nodes .+ neurons_per_layer

        connectivity_matrix[current_nodes, next_nodes] .= true
        connectivity_matrix[next_nodes, current_nodes] .= true
    end

    return connectivity_matrix
end

function AllToAllTNN(
    n_layers::Int,
    neurons_per_layer::Int,
    Wᵢⱼ::AbstractMatrix{Float64}; # external coupling
    J₂ = 1.0, J₃ = 0.0, J₄ = 1.0
)

    @assert n_layers ≥ 1 && neurons_per_layer ≥ 1 "Layer and neuron counts must be positive"
    @assert size(Wᵢⱼ, 2) == neurons_per_layer "External coupling matrix must have the same number of columns as neurons in first layer"

    n_neurons = n_layers * neurons_per_layer
    neurons = [QuarticNeuron(J₂, J₃, J₄) for _ in 1:n_neurons]
    biases = randn(n_neurons)

    connectivity_matrix = all_to_all_connectivity_matrix(n_layers, neurons_per_layer)

    Jᵢⱼ = randn(n_neurons, n_neurons)
    Jᵢⱼ = 0.5 * (Jᵢⱼ + transpose(Jᵢⱼ))
    Jᵢⱼ .*= connectivity_matrix

    return ThermodynamicNeuralNetwork(neurons, biases, Jᵢⱼ, Wᵢⱼ, connectivity_matrix)
end


# Eqn 5
function V_internal(tnn::TNN, x::AbstractVector{T}) where T
    @assert length(x) == n_neurons(tnn) "State vector length must match the neuron count"

    V_local = mapreduce(potential_energy, +, neurons(tnn), x, bias(tnn))
    V_coupling = 0.5 * dot(x, internal_coupling(tnn) * x)
    return V_local + V_coupling
end

# Eqn 6
function V_external(tnn::TNN, x::AbstractVector{T}, I::AbstractVector{T}) where T
    Wᵢⱼ = external_coupling(tnn)
    @assert length(x) == n_neurons(tnn) "State vector length must match the neuron count"
    @assert length(I) == size(Wᵢⱼ, 1) "Input vector length must match the number of rows in the external coupling matrix"

    x_first = view(x, 1:size(Wᵢⱼ, 2))
    return dot(I, Wᵢⱼ * x_first)
end