export train

function mse_loss(tnn, sim, X₀, I, y_target; seed = nothing, trajectory_seeds = nothing, return_predictions = false)

    y_pred = _predict!(tnn, sim, X₀, I; seed, trajectory_seeds)

    yp_sz = size(y_pred) 
    yt_sz = size(y_target)
    # Both should be (n_outputs, n_points)
    @assert yp_sz == yt_sz "Prediction $(yp_sz) and y_target $(yt_sz) shapes must match for MSE loss"

    
    value = sum(abs2, y_pred .- y_target) / yp_sz[2]
    return return_predictions ? (value, y_pred) : value
end

function model_info(tnn::TNN)
    trainable = Optimisers.trainables(tnn)

    for (i, p) in enumerate(trainable)
        println("Parameter $i: $(size(p)), $(typeof(p))")
    end

    println("Total trainable parameters: ", sum(length, trainable))

end


"""
    train(n_inputs, n_layers, neurons_per_layer, n_outputs, sim_settings, z, targets; kwargs...)

Train using input columns `z` of shape `(n_inputs, n_points)` and corresponding
target columns of shape `(n_outputs, n_points)`. Vectors of lengths `n_inputs`
and `n_outputs` represent a single point. Both arguments must have the same concrete array type.
Training supports CPU vectors/matrices, reset or serial sampling, and fixed-step EM.
Set `loss=categorical_crossentropy` for classification with one-hot target columns;
the default is `loss=mse_loss`. Predictions and snapshots remain raw averaged
scores. Apply `class_probabilities` afterward when probabilities are needed.
This trainer uses the complete supplied batch on each update.
`nthreads=Threads.nthreads()` limits concurrent reset trajectories and AD workers.
Set `nthreads=1` for serial execution, or start Julia with `--threads=auto` to
make more threads available. `trajectory_batch_size` defaults to `nthreads`;
a smaller value further limits concurrency and memory. Each worker reuses its
own ReverseDiff tape across trajectories and updates, so tape memory scales
with the worker count. SerialMode uses one worker for its continuous trajectory.
A fresh `ReplayPlan` is drawn from the caller's RNG each iteration; seed that
RNG with `Random.seed!` before training for reproducibility.

By default, return `(tnn, losses)`. Set `saveat` to a sorted collection of unique
training iterations in `0:maxiters` to also return prediction snapshots:
`(tnn, losses, saved)`, where `saved.iterations` and `saved.predictions` correspond
element by element. Each prediction is an `(n_outputs, n_points)` matrix evaluated
using the simulation's sampling scheme, after the requested number of optimizer updates;
iteration 0 is the untrained model. Snapshots are retained in memory.

Intermediate snapshots reuse the next training pass's predictions and noise,
so their chosen loss equals `losses[iteration + 1]`. Saving `maxiters` requires one extra
forward evaluation of the final model with fresh noise. No extra solves occur
for intermediate snapshots. An empty `saveat` returns empty snapshot vectors.
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
    loss=mse_loss,
    maxiters::Int = 1000,
    saveat = nothing,
    nthreads::Int=Threads.nthreads(),
    trajectory_batch_size::Int=nthreads,
) where {FT <: AbstractFloat, SM <: SamplingMode, T <: AbstractVecOrMat{FT}}

    maxiters >= 0 || throw(ArgumentError("maxiters must be nonnegative"))
    SM in (ResetMode, SerialMode) || throw(ArgumentError("unsupported sampling mode"))
    _check_loss(loss)
    sim_settings.integrator isa SDE.EM || throw(ArgumentError("training requires fixed-step EM()"))
    sim_settings.M > 0 || throw(ArgumentError("training requires positive trajectories"))
    1 <= nthreads <= Threads.nthreads() ||
        throw(ArgumentError("nthreads must be between 1 and Threads.nthreads(); start Julia with more threads to raise this limit"))
    trajectory_batch_size > 0 || throw(ArgumentError("trajectory_batch_size must be positive"))
    batch_size = min(trajectory_batch_size, nthreads, sim_settings.M)
    threaded = SM === ResetMode && batch_size > 1
    iterations = saveat === nothing ? Int[] : collect(saveat)
    all(i -> i isa Integer && 0 <= i <= maxiters, iterations) ||
        throw(ArgumentError("saveat must contain integer iterations in 0:maxiters"))
    issorted(iterations) && allunique(iterations) ||
        throw(ArgumentError("saveat must be sorted and contain no duplicates"))
    iterations = Int.(iterations)

    size(z, 1) == n_inputs || throw(DimensionMismatch("z must have $n_inputs rows; got $(size(z, 1))"))
    size(targets, 1) == n_outputs || throw(DimensionMismatch("targets must have $n_outputs rows; got $(size(targets, 1))"))
    n_points = size(z, 2)
    size(targets, 2) == n_points || throw(DimensionMismatch("z and targets must have the same number of points; got $n_points and $(size(targets, 2))"))
    n_points > 0 || throw(ArgumentError("training data must contain at least one point"))

    if z isa AbstractVector
        z = reshape(z, n_inputs, 1)
        targets = reshape(targets, n_outputs, 1)
    end
    z isa Matrix{FT} && targets isa Matrix{FT} ||
        throw(ArgumentError("training requires CPU vectors or matrices"))
    loss === categorical_crossentropy && _check_categorical_targets(zeros(FT, size(targets)), targets)

    n_neurons = neurons_per_layer * n_layers
    params = TNNParameters(n_inputs, n_layers, neurons_per_layer, n_outputs, z)
    tnn = TNN(params, n_neurons, n_layers)

    opt_state = Optimisers.setup(optimizer, tnn)

    # Initialize neuron state
    X₀ = similar(z, n_neurons, n_points)
    X₀ .= FT(0.0)

    # Make sure its right floating point type
    sim_settings = SimulationSettings(sim_settings, FT)

    cache = if maxiters == 0
        nothing
    elseif threaded
        ThreadedReplayGradientCache(tnn, sim_settings, z; batch_size, workers=nthreads)
    else
        ReplayGradientCache(tnn, sim_settings, z)
    end

    loss_buffer = Vector{FT}(undef, maxiters)
    saved_predictions = Vector{AbstractMatrix{FT}}(undef, length(iterations))
    next_save = 1

    model_info(tnn)
    progress = Progress(maxiters; desc="Training ", showspeed=true)
    update!(progress, 0; force=true)

    for i in 1:maxiters
        plan = ReplayPlan(sim_settings; seed=rand(UInt64))
        result = replay_loss_gradient(tnn, sim_settings, z, targets, plan; loss, cache, initial_state=X₀)
        value = result.loss
        if !isempty(iterations)
            if next_save <= length(iterations) && iterations[next_save] == i - 1
                saved_predictions[next_save] = copy(result.predictions)
                next_save += 1
            end
        end

        opt_state, tnn = Optimisers.update(
            opt_state, tnn, result.gradient
        )
        enforce_connectivity!(tnn)

        loss_buffer[i] = value
        next!(progress; showvalues=[(:iteration, i), (:loss, value)])
    end

    if next_save <= length(iterations)
        # Only the final requested snapshot cannot be reused from a training pass.
        saved_predictions[next_save] = if threaded
            plan = ReplayPlan(sim_settings; seed=rand(UInt64))
            first(_threaded_forward(tnn, sim_settings, z, X₀, plan.seeds, batch_size; check_replay=false))
        else
            predict(tnn, sim_settings, z)
        end
    end

    saved = (; iterations, predictions=saved_predictions)
    return saveat === nothing ? (tnn, loss_buffer) : (tnn, loss_buffer, saved)
end

