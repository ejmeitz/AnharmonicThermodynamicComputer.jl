struct Layer{N}
    nodes::SVector{N, ThermodynamicNeuron}
    Jᵢⱼ::AbstractMatrix{Float64}
    bᵢ::AbstractVector{Float64}
end

# Builds layers of identical neurons
function Layer(neuron, n_neurons::Int)
    # Probably a better way to represent the couplings
    # Paper implies they are symmetric, and
    # that there is no self-coupling
    Jᵢⱼ = randn(n_neurons, n_neurons)
    Jᵢⱼ = 0.5 * (Jᵢⱼ + Jᵢⱼ') # make symmetric
    Jᵢⱼ[diagind(Jᵢⱼ)] .= 0.0 # Remove self-couplings
    bᵢ = randn(n_neurons)
    neurons = SVector{n_neurons}(fill(neuron, n_neurons))
    return Layer(neurons, Jᵢⱼ, bᵢ)
end

nodes(layer::Layer) = layer.nodes
bias(layer::Layer) = layer.bᵢ
coupling(layer::Layer) = layer.Jᵢⱼ
n_neurons(::Layer{N}) where N = N

# Eqn 5
function V_internal(x, layer::Layer)
    V_local = mapreduce(potential_energy, +, nodes(layer), x, bias(layer))
    V_coupling = 0.5 * dot(x, coupling(layer) * x)
    return V_local + V_coupling
end
