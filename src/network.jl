export TNN, TNNParameters, enforce_connectivity!

abstract type Connectivity end
struct AllToAll <: Connectivity end

############################

#! need to figure out how to store connectivity in here
struct TNNParameters{
    FT <: AbstractFloat,
    V <: AbstractVector{FT},
    M <: AbstractMatrix{FT}
}
    bᵢ::V               # (n_neurons,)
    Jᵢⱼ::M              # Internal coupling: (n_neurons, n_neurons)
    Wᵢⱼ::M              # External coupling: (n_inputs, n_first_layer_neurons)
    fᵢ::M               # (n_outputs, n_last_layer_neurons)
    J_scale::M
    W_scale::M
end

# Lets me use structs and tells Optimisers.jl
# which children to look in for trainable parameters
Functors.@functor TNNParameters (bᵢ, Jᵢⱼ, Wᵢⱼ, fᵢ)


biases(tnnp::TNNParameters) = tnnp.bᵢ
internal_coupling(tnnp::TNNParameters) = tnnp.Jᵢⱼ
external_coupling(tnnp::TNNParameters) = tnnp.Wᵢⱼ
output_coupling(tnnp::TNNParameters) = tnnp.fᵢ
n_inputs(tnnp::TNNParameters) = size(external_coupling(tnnp), 1)
n_outputs(tnnp::TNNParameters) = size(output_coupling(tnnp), 1)
n_first_layer_neurons(tnnp::TNNParameters) = size(external_coupling(tnnp), 2)
n_last_layer_neurons(tnnp::TNNParameters) = size(output_coupling(tnnp), 2)

#! only support this for now
connectivity(::TNNParameters) = AllToAll

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

    return J_scale, W_scale
end

float_type(::Type{<:AbstractArray{T}}) where T = T

"""
    TNNParameters(n_inputs, n_layers, neurons_per_layer, n_outputs, array_prototype;
                  σ=0.1, coupling_σ=1.0)

Initialize Gaussian parameters with standard deviation `coupling_σ` for W and
allowed symmetric J couplings, and `σ` for biases and the output readout.
"""
function TNNParameters(
    n_inputs::Int,
    n_layers::Int,
    neurons_per_layer::Int,
    n_outputs::Int,
    array_prototype::AT;
    σ = 0.1,
    coupling_σ = 1.0,
) where {AT <: AbstractArray}

    FT = float_type(AT)
    σ = FT(σ)
    coupling_σ = FT(coupling_σ)

    n_neurons = n_layers * neurons_per_layer
    
    Wᵢⱼ = similar(array_prototype, n_inputs, neurons_per_layer)
    randn!(Wᵢⱼ)
    Wᵢⱼ .*= coupling_σ

    bᵢ = similar(Wᵢⱼ, n_neurons)
    randn!(bᵢ)
    bᵢ .*= σ

    connectivity_matrix = similar(Wᵢⱼ, Bool, n_neurons, n_neurons)
    all_to_all_connectivity_matrix!(connectivity_matrix, n_layers, neurons_per_layer)

    Jᵢⱼ = similar(Wᵢⱼ, n_neurons, n_neurons)
    randn!(Jᵢⱼ)
    Jᵢⱼ = FT(coupling_σ / sqrt(FT(2))) .* (Jᵢⱼ .+ transpose(Jᵢⱼ))
    Jᵢⱼ .*= connectivity_matrix

    fᵢ = similar(Wᵢⱼ, n_outputs, neurons_per_layer)
    randn!(fᵢ)
    fᵢ .*= σ

    J_scale, W_scale = mutation_scales(connectivity_matrix, Wᵢⱼ)

    return TNNParameters(bᵢ, Jᵢⱼ, Wᵢⱼ, fᵢ, J_scale, W_scale)
end


############################

struct TNN{
    FT,
    C <: Connectivity,
    N <: AbstractVector{<:ThermodynamicNeuron{FT}},
    P <: TNNParameters{FT},
}
    neurons::N
    params::P
    n_layers::Int
end

# Lets me use structs and tells Optimisers.jl
# which children to look in for trainable parameters
function Functors.functor(::Type{<:TNN{FT,C}}, tnn) where {FT,C}
    children = (; params = tnn.params)
    # C is not a field, so the default reconstruction cannot infer it.
    reconstruct = p -> TNN{FT,C,typeof(tnn.neurons),typeof(p.params)}(
        tnn.neurons, p.params, tnn.n_layers
    )
    return children, reconstruct
end

params(tnn::TNN) = tnn.params
neurons(tnn::TNN) = tnn.neurons
bias(tnn::TNN) = biases(params(tnn))
internal_coupling(tnn::TNN) = internal_coupling(params(tnn))
external_coupling(tnn::TNN) = external_coupling(params(tnn))
output_coupling(tnn::TNN) = output_coupling(params(tnn))

n_neurons(tnn::TNN) = length(neurons(tnn))
Base.length(tnn::TNN) = n_neurons(tnn)
n_layers(tnn::TNN) = tnn.n_layers
n_first_layer_neurons(tnn::TNN) = n_first_layer_neurons(params(tnn))
n_last_layer_neurons(tnn::TNN) = n_last_layer_neurons(params(tnn))

# Project each optimizer update back onto the current AllToAll constraints.
# J_scale's nonzero pattern marks allowed edges; its magnitudes are not applied.
function enforce_connectivity!(tnn::TNN{FT,AllToAll}) where FT
    p = params(tnn)
    J = p.Jᵢⱼ
    for j in axes(J, 2)
        J[j, j] = zero(FT)
        for i in 1:j-1
            coupling = iszero(p.J_scale[i, j]) ? zero(FT) : (J[i, j] + J[j, i]) / 2
            J[i, j] = J[j, i] = coupling
        end
    end
    return tnn
end

function TNN(
    parameters::TNNParameters{FT},
    n_neurons::Int,
    n_layers::Int;
    J₂ = one(FT), J₃ = zero(FT), J₄ = one(FT)
) where {FT}

    J₂, J₃, J₄ = FT(J₂), FT(J₃), FT(J₄)

    # Initialize things in a way that is agnostic
    # to the array backend (i.e. Array or CuArray)
    neuron = QuarticNeuron(FT(J₂), FT(J₃), FT(J₄))
    neurons = similar(output_coupling(parameters), typeof(neuron), n_neurons)
    fill!(neurons, neuron)

    # Just hardcode AllToAll for now
    return TNN{FT, AllToAll, typeof(neurons), typeof(parameters)}(
        neurons,
        parameters,
        n_layers
    )
end

function TNN(
    n_inputs::Int,
    n_layers::Int,
    neurons_per_layer::Int,
    n_outputs::Int,
    array_prototype::AbstractArray{FT};
    J₂ = one(FT), J₃ = zero(FT), J₄ = one(FT)
) where {FT <: AbstractFloat}
    @assert n_layers ≥ 1 && neurons_per_layer ≥ 1 "Layer and neuron counts must be positive"
    params = TNNParameters(n_inputs, n_layers, neurons_per_layer, n_outputs, array_prototype)
    n_neurons = n_layers * neurons_per_layer
    return TNN(params, n_neurons, n_layers; J₂=J₂, J₃=J₃, J₄=J₄)
end


#############################


# Eqn 5
function V_internal(tnn::TNN{FT}, x::AbstractVector{FT}) where FT
    V_local = mapreduce(potential_energy, +, neurons(tnn), x, bias(tnn))
    V_coupling = FT(0.5) * dot(x, internal_coupling(tnn) * x)
    return V_local + V_coupling
end

function V_internal(tnn::TNN{FT}, X::AbstractVecOrMat{FT}) where FT
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

function add_F_internal!(F::T, tnn::TNN{FT}, X::T) where {FT, T <: AbstractVecOrMat{FT}}
    return _add_F_internal!(F, neurons(tnn), params(tnn), X)
end

# This variant is needed for AD to work
function _add_F_internal!(F, neuron_data, p, X)
    F .+= force.(neuron_data, X, p.bᵢ) # F_local
    F .-= p.Jᵢⱼ * X # F_coupling
    return F
end

# Eqn 6
function V_external(tnn::TNN{FT}, X::T, I::T) where {FT, T <: AbstractVecOrMat{FT}}
    X_first = view(X, 1:n_first_layer_neurons(tnn), :)
    return sum(I .* (external_coupling(tnn) * X_first); dims=1)
end

function add_F_external!(F::T, tnn::TNN{FT}, I::T) where {FT, T <: AbstractVecOrMat{FT}}
    return _add_F_external!(F, params(tnn), I)
end

# This variant is needed for AD to work
function _add_F_external!(F, p, I)
    F_first = view(F, 1:size(p.Wᵢⱼ, 2), :)
    F_first .-= transpose(p.Wᵢⱼ) * I
    return F
end

potential_energy(tnn::TNN{FT}, x::T, I::T) where {FT, T <: AbstractVecOrMat{FT}} = V_internal(tnn, x) .+ V_external(tnn, x, I)

function force!(F::T, tnn::TNN{FT}, x::T, I::T) where {FT, T <: AbstractVecOrMat{FT}}
    add_F_internal!(F, tnn, x)
    add_F_external!(F, tnn, I)
    return F
end
