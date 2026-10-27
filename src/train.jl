# https://docs.sciml.ai/SciMLSensitivity/stable/examples/sde/optimization_sde/#Example-3:-Controlling-SDEs-to-an-objective

function mse_loss(tnn, sim, X₀, I, y_target, sensealg; seed = nothing)

    y_pred = _predict!(tnn, sim, X₀, I, sensealg; seed)

    yp_sz = size(y_pred) 
    yt_sz = size(y_target)
    # Both should be (n_outputs, n_points)
    @assert yp_sz == yt_sz "Prediction $(yp_sz) and y_target $(yt_sz) shapes must match for MSE loss"

    
    return sum(abs2, y_pred .- y_target) / yp_sz[2]
end

function model_info(tnn::TNN)
    trainable = Optimisers.trainables(tnn)

    for (i, p) in enumerate(trainable)
        println("Parameter $i: $(size(p)), $(typeof(p))")
    end

    println("Total trainable parameters: ", sum(length, trainable))

end


#! right now hard coded for MSE, need to abstract to other losses 
"""
    train(n_inputs, n_layers, neurons_per_layer, n_outputs, sim_settings, z, targets; kwargs...)

Train using input columns `z` of shape `(n_inputs, n_points)` and corresponding
target columns of shape `(n_outputs, n_points)`. Vectors of lengths `n_inputs`
and `n_outputs` represent a single point. Both arguments must have the same concrete array type.
"""
function train(
    n_inputs::Int,
    n_layers::Int,
    neurons_per_layer::Int,
    n_outputs::Int,
    sim_settings::SimulationSettings{SM},
    z::T, # external forcing
    targets::T;
    optimizer=Optimisers.Adam(0.01),
    sensealg = SciMLSensitivity.ReverseDiffAdjoint(),
    maxiters::Int = 1000,
) where {FT <: AbstractFloat, SM <: SamplingMode, T <: AbstractVecOrMat{FT}}

    size(z, 1) == n_inputs || throw(DimensionMismatch("z must have $n_inputs rows; got $(size(z, 1))"))
    size(targets, 1) == n_outputs || throw(DimensionMismatch("targets must have $n_outputs rows; got $(size(targets, 1))"))
    n_points = size(z, 2)
    size(targets, 2) == n_points || throw(DimensionMismatch("z and targets must have the same number of points; got $n_points and $(size(targets, 2))"))
    n_points > 0 || throw(ArgumentError("training data must contain at least one point"))

    if z isa AbstractVector
        z = reshape(z, n_inputs, 1)
        targets = reshape(targets, n_outputs, 1)
    end

    n_neurons = neurons_per_layer * n_layers
    params = TNNParameters(n_inputs, n_layers, neurons_per_layer, n_outputs, z)
    tnn = TNN(params, n_neurons, n_layers)

    opt_state = Optimisers.setup(optimizer, tnn)

    # Initialize neuron state
    X₀ = similar(z, n_neurons, n_points)
    X₀ .= FT(0.0)

    # Make sure its right floating point type
    sim_settings = SimulationSettings(sim_settings, FT)

    loss(tnn) = mse_loss(
        tnn, sim_settings, X₀, z, targets, sensealg
    )

    loss_buffer = Vector{FT}(undef, maxiters)

    model_info(tnn)

    for i in 1:maxiters
        value, grads = Zygote.withgradient(loss, tnn)

        opt_state, tnn = Optimisers.update(
            opt_state, tnn, only(grads)
        )

        loss_buffer[i] = value
    end

    return tnn, loss_buffer
end

