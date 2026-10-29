export ReplayPlan, replay_mse_gradient, replay_loss_gradient

"""
    ReplayPlan(trajectories; seed)
    ReplayPlan(sim; seed)

Explicit per-trajectory seeds for the two-pass gradient prototype. Uses a local
Xoshiro RNG, without changing the caller's RNG. Reuse a plan to reproduce the
same noise; use a new master seed for fresh noise on another training update.
Reproducibility assumes the same Julia/package versions, solver, timestep,
state shape, and trajectory seed. It is not a cross-version RNG guarantee.
"""
struct ReplayPlan
    seeds::Vector{UInt64}
end

function ReplayPlan(trajectories::Integer; seed::Integer)
    trajectories > 0 || throw(ArgumentError("trajectories must be positive"))
    rng = Random.Xoshiro(seed)
    return ReplayPlan([rand(rng, UInt64) for _ in 1:trajectories])
end

ReplayPlan(sim::SimulationSettings; seed::Integer) = ReplayPlan(_trajectory_count(sim); seed)

function _single_trajectory_settings(sim::SimulationSettings{ResetMode,FT}) where {FT}
    return SimulationSettings{ResetMode,FT,typeof(sim.integrator)}(
        sim.integrator, sim.Δt, sim.t_final, sim.noise_scale, sim.μ, 1, copy(sim.saveat),
    )
end

_single_trajectory_settings(sim::SimulationSettings{SerialMode,FT}) where {FT} = SimulationSettings(sim, FT)

# Scan every saved Brownian value, retaining only a small digest per trajectory.
# Tests also compare the full histories exactly, so correctness does not rest
# solely on this runtime hash check. This hook runs outside differentiation.
function _noise_fingerprint(sol)
    fingerprint = UInt(0)
    for trajectory in sol.u
        W = trajectory.W
        fingerprint = hash(length(W.t), fingerprint)
        for (t, values) in zip(W.t, W.W)
            fingerprint = hash(t, fingerprint)
            for value in values
                fingerprint = hash(value, fingerprint)
            end
        end
    end
    return fingerprint
end

"""
    replay_loss_gradient(tnn, sim, inputs, targets, plan; loss=mse_loss,
                        initial_state=nothing, check_replay=true, progress=false,
                        cache=nothing)

Two-pass gradient of loss **of the mean prediction**, using one reusable
ReverseDiff tape. Supports `mse_loss` and `categorical_crossentropy`, CPU matrices,
reset or serial sampling, and fixed-step Euler–Maruyama. In serial mode, the mean
includes every observation in `sim.saveat` on one continuous trajectory. Use
`ReplayPlan(sim; seed)` to obtain the correct number of trajectory seeds.
This is the gradient implementation used by `train`.

Pass 1 averages predictions without AD. Pass 2 recreates each trajectory's RNG
from `plan.seeds`, applies the common output cotangent, and accumulates its
parameter gradient before reusing the tape for the next trajectory. Parameters and
inputs must remain unchanged during both passes.

Returns `(; loss, predictions, gradient, seeds, noise_fingerprints)`. The gradient
has the structure expected by `Optimisers.update!`. With `check_replay=true`, a
digest of every Brownian value is compared across passes and a mismatch throws.
Disabling this check skips the digest scan, not the deterministic replay.

Omitting `cache` constructs one tape for this call and reuses it across all
trajectories. Pass a `ReplayGradientCache(tnn, sim, inputs)` to also reuse the
tape and its gradient buffers across calls. Forward solves still allocate.
See `ReplayGradientCache` for fixed-configuration restrictions.
Set `threaded=true, trajectory_batch_size=10` to construct a
`ThreadedReplayGradientCache` instead, or pass that cache explicitly to reuse it.
Threaded replay requires reset sampling and multiple Julia threads. The supplied
cache determines execution when `cache` is given.
"""
function replay_loss_gradient(
    tnn::TNN{FT},
    sim::SimulationSettings,
    inputs::Matrix{FT},
    targets::Matrix{FT},
    plan::ReplayPlan;
    loss=mse_loss,
    initial_state=nothing,
    check_replay::Bool=true,
    progress::Bool=false,
    cache=nothing,
    threaded::Bool=false,
    trajectory_batch_size::Int=10,
) where {FT}
    sim.integrator isa SDE.EM || throw(ArgumentError("the replay prototype requires EM()"))
    sim.M > 0 || throw(ArgumentError("sim.M must be positive"))
    length(plan.seeds) == _trajectory_count(sim) || throw(DimensionMismatch("plan needs one seed per independent trajectory; use ReplayPlan(sim; seed)"))
    _check_loss(loss)
    isfinite(sim.Δt) && sim.Δt > 0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(sim.t_final) && sim.t_final > 0 || throw(ArgumentError("t_final must be finite and positive"))
    size(inputs, 1) == n_inputs(params(tnn)) || throw(DimensionMismatch("incorrect input row count"))
    K = size(inputs, 2)
    K > 0 || throw(ArgumentError("inputs must contain at least one point"))
    size(targets) == (n_outputs(params(tnn)), K) || throw(DimensionMismatch("incorrect target shape"))
    loss === categorical_crossentropy && _check_categorical_targets(zeros(FT, size(targets)), targets)
    initial = initial_state === nothing ? zeros(FT, n_neurons(tnn), K) : copy(initial_state)
    initial isa Matrix{FT} || throw(ArgumentError("initial_state must be a Matrix{$FT}"))
    size(initial) == (n_neurons(tnn), K) || throw(DimensionMismatch("incorrect initial_state shape"))

    threaded && _check_threaded_settings(sim, trajectory_batch_size)
    if threaded && cache !== nothing && !(cache isa ThreadedReplayGradientCache)
        throw(ArgumentError("threaded=true requires a ThreadedReplayGradientCache"))
    end
    workspace = if cache !== nothing
        cache
    elseif threaded
        ThreadedReplayGradientCache(tnn, sim, inputs; batch_size=trajectory_batch_size)
    else
        ReplayGradientCache(tnn, sim, inputs)
    end
    return _cached_replay_loss_gradient(workspace, tnn, SimulationSettings(sim, FT),
        inputs, targets, plan, initial; loss, check_replay, progress)
end

"""MSE convenience wrapper for [`replay_loss_gradient`](@ref), in either sampling mode."""
replay_mse_gradient(args...; kwargs...) = replay_loss_gradient(args...; loss=mse_loss, kwargs...)
