export ReplayGradientCache

"""
    ReplayGradientCache(tnn, sim, inputs)

Record one reusable ReverseDiff tape for the fixed-step EM replay prototype.
Pass it to `replay_mse_gradient(...; cache)` to reuse the tape and derivative
buffers across trajectories and training updates. Tape compilation is not used.

Parameters, input values, initial states, noise increments, and the output
cotangent are explicit tape inputs. They may change between calls, as may the
number of trajectories. Array shapes/types, neuron coefficients, timestep,
final time, mobility, and temperature must remain fixed; otherwise rebuild.
CPU matrices, QuarticNeuron (including its harmonic case), and ResetMode/EM
only. A cache is mutable workspace and must not
be used concurrently. Returned results own their storage and survive reuse.

The SDE solver still supplies each trajectory's Brownian history. The tape
replays EM using increments from that history, with array-level operations.
Floating-point operation order can differ from the solver at roundoff level.
With `check_replay=true`, both Brownian fingerprints and each tape prediction
are checked against the SDE solves. This reduces tape construction allocations,
but forward solves and some replay operations still allocate.
"""
struct ReplayGradientCache{T,D}
    tape::T
    data::D
end

# Pure, fixed-shape arithmetic: no random draws or data-dependent branches are
# recorded. Array operations retain ReverseDiff's array-level differentiation.
function _em_replay_output(layout, sim, steps, coefficients, values, noise, inputs, initial)
    p = _parameter_views(layout, values)
    J2, J3, J4, first_layer = coefficients
    N, K = size(initial)
    external = first_layer * (transpose(p.Wᵢⱼ) * inputs)
    X = initial
    for s in eachindex(steps)
        local_force = p.bᵢ .- X .* (2 .* J2 .+ X .* (3 .* J3 .+ 4 .* J4 .* X))
        F = ((local_force .- p.Jᵢⱼ * X) .- external) .* sim.μ
        X = (X .+ steps[s] .* F) .+ sim.noise_scale .* reshape(view(noise, :, s), N, K)
    end
    return p.fᵢ * X[end-size(p.fᵢ, 2)+1:end, :]
end

function _copy_noise_increments!(noise, times, sol)
    W = only(sol.u).W
    isequal(W.t, times) || error("EM time grid changed; rebuild ReplayGradientCache")
    for s in axes(noise, 2)
        @views noise[:, s] .= W.W[s + 1] .- W.W[s]
    end
    return nothing
end

function ReplayGradientCache(tnn::TNN{FT}, sim::SimulationSettings{ResetMode}, inputs::Matrix{FT}) where {FT}
    sim.integrator isa SDE.EM || throw(ArgumentError("the tape cache requires EM()"))
    all(n -> n isa QuarticNeuron, neurons(tnn)) ||
        throw(ArgumentError("the tape cache requires QuarticNeuron neurons"))
    isfinite(sim.Δt) && sim.Δt > 0 && isfinite(sim.t_final) && sim.t_final > 0 ||
        throw(ArgumentError("dt and t_final must be finite and positive"))
    size(inputs, 1) == n_inputs(params(tnn)) && size(inputs, 2) > 0 ||
        throw(DimensionMismatch("incorrect or empty inputs"))
    layout = deepcopy(tnn)
    one_sim = _single_trajectory_settings(SimulationSettings(sim, FT))
    initial = zeros(FT, n_neurons(tnn), size(inputs, 2))
    # Discover the solver's exact saved grid, including a shortened final step.
    history = Ref{Any}()
    _predict!(tnn, one_sim, initial, inputs; seed=0,
        trajectory_seeds=UInt64[0], noise_observer=sol -> (history[] = only(sol.u).W))
    times = copy(history[].t)
    steps = diff(times)
    noise = Matrix{FT}(undef, length(initial), length(steps))
    for s in eachindex(steps)
        @views noise[:, s] .= history[].W[s + 1] .- history[].W[s]
    end
    first_layer = zeros(FT, n_neurons(tnn), n_first_layer_neurons(tnn))
    for i in axes(first_layer, 2)
        first_layer[i, i] = one(FT)
    end
    coefficients = (getproperty.(neurons(layout), :J₂),
                    getproperty.(neurons(layout), :J₃), getproperty.(neurons(layout), :J₄), first_layer)
    tape_inputs = (_pack_parameters(tnn), noise, copy(inputs), initial,
                   zeros(FT, n_outputs(params(tnn)), size(inputs, 2)))
    objective = (values, increments, I, X0, cotangent) ->
        sum(_em_replay_output(layout, one_sim, steps, coefficients, values, increments, I, X0) .* cotangent)
    tape = ReverseDiff.GradientTape(objective, tape_inputs)
    derivatives = map(zero, tape_inputs)
    data = (; layout, sim=one_sim, times, steps, coefficients, tape_inputs, derivatives)
    return ReplayGradientCache(tape, data)
end

function _check_replay_cache(cache::ReplayGradientCache, tnn, sim, inputs)
    d = cache.data
    size(inputs) == size(d.tape_inputs[3]) && eltype(inputs) == eltype(d.tape_inputs[3]) ||
        throw(DimensionMismatch("input shape/type changed; rebuild ReplayGradientCache"))
    all(name -> size(getproperty(params(tnn), name)) == size(getproperty(params(d.layout), name)),
        (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)) && isequal(neurons(tnn), neurons(d.layout)) ||
        throw(ArgumentError("network structure or neuron coefficients changed; rebuild ReplayGradientCache"))
    all(name -> isequal(getproperty(sim, name), getproperty(d.sim, name)),
        (:Δt, :t_final, :noise_scale, :μ)) ||
        throw(ArgumentError("simulation configuration changed; rebuild ReplayGradientCache"))
    return nothing
end

function _cached_replay_mse_gradient(cache::ReplayGradientCache, tnn::TNN{FT}, sim,
    inputs, targets, plan, initial; check_replay, progress) where {FT}
    _check_replay_cache(cache, tnn, sim, inputs)
    d = cache.data
    values, noise, I, X0, cotangent = d.tape_inputs
    values .= _pack_parameters(tnn)
    I .= inputs
    X0 .= initial
    seeds = copy(plan.seeds)
    predictions = zeros(FT, size(targets))
    fingerprints = check_replay ? Vector{UInt}(undef, sim.M) : nothing
    meter = Progress(2 * sim.M; desc="Cached replay ", showspeed=true, enabled=progress)
    update!(meter, 0; force=true)
    for i in eachindex(seeds)
        observer = check_replay ? sol -> (fingerprints[i] = _noise_fingerprint(sol)) : nothing
        predictions .+= _predict!(tnn, d.sim, X0, I;
            seed=0, trajectory_seeds=view(seeds, i:i), noise_observer=observer)
        next!(meter)
    end
    predictions ./= FT(sim.M)
    residual = predictions .- targets
    loss = sum(abs2, residual) / size(inputs, 2)
    isfinite(loss) || error("replay forward pass produced a non-finite loss")
    cotangent .= (FT(2) / (FT(size(inputs, 2)) * FT(sim.M))) .* residual
    accumulated = zero(values)
    for i in eachindex(seeds)
        observer = sol -> begin
            if check_replay
                _noise_fingerprint(sol) == fingerprints[i] ||
                    error("Brownian replay mismatch for trajectory $i; gradient discarded")
            end
            _copy_noise_increments!(noise, d.times, sol)
        end
        reference = _predict!(tnn, d.sim, X0, I;
            seed=0, trajectory_seeds=view(seeds, i:i), noise_observer=observer)
        ReverseDiff.gradient!(d.derivatives, cache.tape, d.tape_inputs)
        # The derivative with respect to the output cotangent is the prediction,
        # including when that cotangent is zero. Check the actual taped execution.
        if check_replay
            isapprox(last(d.derivatives), reference; rtol=100 * eps(FT), atol=100 * eps(FT)) ||
                error("Cached EM prediction differs from the SDE solve for trajectory $i; gradient discarded")
        end
        accumulated .+= first(d.derivatives)
        next!(meter)
    end
    all(isfinite, accumulated) || error("replay produced a non-finite gradient")
    gradient = (; params=map(copy, _parameter_views(tnn, accumulated)))
    return (; loss, predictions, gradient, seeds, noise_fingerprints=fingerprints)
end
