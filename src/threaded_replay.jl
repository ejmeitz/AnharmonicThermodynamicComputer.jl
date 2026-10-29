export ThreadedReplayGradientCache

"""
    ThreadedReplayGradientCache(tnn, sim, inputs; batch_size=10, workers=Threads.nthreads())

Opt-in reset-mode replay with bounded `EnsembleThreads` solves and concurrent
backpropagation. Start Julia with multiple threads (for example `--threads=4`).
Each worker owns a complete `ReplayGradientCache`; tape memory scales with the
worker count, and saved Brownian history scales with `batch_size`, not total M.
The complete-mean loss and per-trajectory seeds are unchanged. Gradient reduction
follows trajectory order, independently of task scheduling.

Workers are capped by the batch size and trajectory count at construction.
The effective batch size is also capped by `workers`, so both forward solves
and backpropagation respect the requested concurrency.
The same fixed-configuration restrictions as `ReplayGradientCache` apply.
Do not share this mutable workspace between concurrent training calls.
SerialMode has only one continuous trajectory and is not supported here.
"""
struct ThreadedReplayGradientCache{C,G}
    workers::Vector{C}
    batch_size::Int
    gradients::G
end

function _check_threaded_settings(sim, batch_size)
    sampling_mode(sim) === ResetMode || throw(ArgumentError("threaded replay requires ResetMode; SerialMode has one continuous trajectory"))
    batch_size isa Integer && batch_size > 0 || throw(ArgumentError("trajectory batch size must be a positive integer"))
    Threads.nthreads() > 1 || throw(ArgumentError("threaded replay requires multiple Julia threads; restart Julia with --threads=4"))
end

function ThreadedReplayGradientCache(tnn, sim, inputs; batch_size=10, workers=Threads.nthreads())
    _check_threaded_settings(sim, batch_size)
    workers isa Integer && 1 <= workers <= Threads.nthreads() ||
        throw(ArgumentError("workers must be between 1 and Threads.nthreads()"))
    count = min(workers, batch_size, sim.M)
    caches = [ReplayGradientCache(tnn, sim, inputs) for _ in 1:count]
    gradients = zeros(eltype(inputs), length(first(caches).data.tape_inputs[1]), count)
    return ThreadedReplayGradientCache(caches, Int(count), gradients)
end

function _chunk_settings(sim::SimulationSettings{ResetMode,FT}, count) where FT
    return SimulationSettings{ResetMode,FT,typeof(sim.integrator)}(
        sim.integrator, sim.Δt, sim.t_final, sim.noise_scale, sim.μ, count, copy(sim.saveat))
end

function _reset_trajectory_output(tnn, trajectory, n_points)
    X = reshape(trajectory.u[end], n_neurons(tnn), n_points)
    return output_coupling(tnn) * @view(X[end-n_last_layer_neurons(tnn)+1:end, :])
end

_trajectory_fingerprint(trajectory) = _noise_fingerprint((; u=(trajectory,)))

function _threaded_forward(tnn::TNN{FT}, sim, inputs, initial, seeds, batch_size;
    check_replay, meter=nothing) where FT
    predictions = zeros(FT, n_outputs(params(tnn)), size(inputs, 2))
    fingerprints = check_replay ? Vector{UInt}(undef, length(seeds)) : nothing
    for first_index in 1:batch_size:length(seeds)
        indices = first_index:min(first_index + batch_size - 1, length(seeds))
        observer = sol -> begin
            # This callback runs after the ensemble joins, on the calling task.
            for (j, trajectory) in enumerate(sol.u)
                predictions .+= _reset_trajectory_output(tnn, trajectory, size(inputs, 2))
                check_replay && (fingerprints[indices[j]] = _trajectory_fingerprint(trajectory))
            end
        end
        _predict!(tnn, _chunk_settings(sim, length(indices)), initial, inputs;
            seed=0, trajectory_seeds=view(seeds, indices), noise_observer=observer,
            ensemblealg=SDE.EnsembleThreads())
        meter === nothing || update!(meter, last(indices))
    end
    predictions ./= length(seeds)
    return predictions, fingerprints
end

function _cached_replay_loss_gradient(pool::ThreadedReplayGradientCache, tnn::TNN{FT}, sim,
    inputs, targets, plan, initial; loss, check_replay, progress) where FT
    _check_threaded_settings(sim, pool.batch_size)
    base = first(pool.workers)
    _check_replay_cache(base, tnn, sim, inputs)
    values = _pack_parameters(tnn)
    seeds = copy(plan.seeds)
    M = length(seeds)
    meter = Progress(2M; desc="Threaded replay ", showspeed=true, enabled=progress)
    update!(meter, 0; force=true)
    predictions, fingerprints = _threaded_forward(tnn, sim, inputs, initial, seeds, pool.batch_size;
        check_replay, meter)
    cotangent = zero(predictions)
    value = _loss_and_cotangent!(cotangent, loss, predictions, targets)
    isfinite(value) || error("threaded forward pass produced a non-finite loss")
    cotangent ./= M
    for worker in pool.workers
        p, noise, I, X0, c = worker.data.tape_inputs
        p .= values
        I .= inputs
        X0 .= initial
        c .= cotangent
    end
    accumulated = zero(values)
    for first_index in 1:pool.batch_size:M
        indices = first_index:min(first_index + pool.batch_size - 1, M)
        history = Ref{Any}()
        _predict!(tnn, _chunk_settings(sim, length(indices)), initial, inputs;
            seed=0, trajectory_seeds=view(seeds, indices), noise_observer=sol -> (history[] = sol),
            ensemblealg=SDE.EnsembleThreads())
        sol = history[]
        active_workers = min(length(pool.workers), length(sol.u))
        @sync for worker_index in 1:active_workers
            # Bind the cache to a task, not threadid(): Julia tasks may migrate.
            worker = pool.workers[worker_index]
            Threads.@spawn begin
                d = worker.data
                for j in worker_index:active_workers:length(sol.u)
                    trajectory = sol.u[j]
                    if check_replay
                        _trajectory_fingerprint(trajectory) == fingerprints[indices[j]] ||
                            error("Brownian replay mismatch for trajectory $(indices[j]); gradient discarded")
                    end
                    _copy_noise_increments!(d.tape_inputs[2], d.times, (; u=(trajectory,)))
                    ReverseDiff.gradient!(d.derivatives, worker.tape, d.tape_inputs)
                    if check_replay
                        reference = _reset_trajectory_output(tnn, trajectory, size(inputs, 2))
                        isapprox(last(d.derivatives), reference; rtol=100eps(FT), atol=100eps(FT)) ||
                            error("Threaded EM prediction differs from the SDE solve")
                    end
                    @views pool.gradients[:, j] .= first(d.derivatives)
                end
            end
        end
        for j in eachindex(sol.u)
            @views accumulated .+= pool.gradients[:, j]
        end
        update!(meter, M + last(indices))
    end
    all(isfinite, accumulated) || error("threaded replay produced a non-finite gradient")
    gradient = (; params=map(copy, _parameter_views(tnn, accumulated)))
    return (; loss=value, predictions, gradient, seeds, noise_fingerprints=fingerprints)
end
