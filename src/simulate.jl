export ResetMode, SerialMode, SimulationSettings, predict

abstract type SamplingMode end

struct ResetMode <: SamplingMode end
struct SerialMode <: SamplingMode end

struct SimulationSettings{
    SM <: SamplingMode,
    FT <: AbstractFloat,
    I
}
    integrator::I
    Δt::FT
    t_final::FT
    noise_scale::FT
    μ::FT
    M::Int
    saveat::Vector{FT}
end

sampling_mode(::SimulationSettings{SM}) where SM = SM

"""
    SimulationSettings(dt, t_final, kBT, μ, M, ResetMode)
    SimulationSettings(dt, t_final, kBT, μ, SerialMode; saveat)

Reset mode averages `M` independent trajectories at `t_final`. Serial mode
averages observations at the explicit times in `saveat` along one continuous
trajectory per input; `M` is inferred from the number of observation times.
Times must be sorted, unique, and within `[0, t_final]`. The solver steps exactly
to these times. Inputs stay fixed throughout the trajectory. Serial observations
can be correlated and need not be equilibrium samples.
"""
function SimulationSettings(Δt, t_final, kBT, μ, M, ::Type{SM}; integrator = SDE.EM(), saveat=nothing) where {SM <: SamplingMode}

    # handle mixed float types
    FT = float(promote_type(
        typeof(Δt),
        typeof(t_final),
        typeof(kBT),
        typeof(μ),
    ))

    SM in (ResetMode, SerialMode) || throw(ArgumentError("unsupported sampling mode"))
    isfinite(Δt) && Δt > 0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(t_final) && t_final > 0 || throw(ArgumentError("t_final must be finite and positive"))
    isfinite(kBT) && kBT >= 0 || throw(ArgumentError("kBT must be finite and nonnegative"))
    isfinite(μ) && μ >= 0 || throw(ArgumentError("mobility must be finite and nonnegative"))
    M isa Integer && M > 0 || throw(ArgumentError("M must be a positive integer"))
    if SM === ResetMode
        saveat === nothing || throw(ArgumentError("simulation saveat is only used with SerialMode"))
        times = FT[]
    else
        saveat isa AbstractVector || throw(ArgumentError("SerialMode requires an explicit vector or range of saveat times"))
        times = FT.(collect(saveat))
        length(times) == M || throw(ArgumentError("M must equal length(saveat) in SerialMode"))
        all(t -> isfinite(t) && 0 <= t <= FT(t_final), times) &&
            issorted(times) && allunique(times) ||
            throw(ArgumentError("saveat times must be finite, sorted, unique, and within [0, t_final]"))
    end

    # Δt factor should come through integrator
    noise_scale = sqrt(2 * kBT * μ)

    return SimulationSettings{SM,FT,typeof(integrator)}(
        integrator,
        FT(Δt),
        FT(t_final),
        FT(noise_scale),
        FT(μ),
        M,
        times,
    )
end

function SimulationSettings(Δt, t_final, kBT, μ, ::Type{SerialMode}; saveat, integrator=SDE.EM())
    saveat isa AbstractVector || throw(ArgumentError("saveat must be a vector or range of times"))
    return SimulationSettings(Δt, t_final, kBT, μ, length(saveat), SerialMode; saveat, integrator)
end

# Reconstruct with specified float type
function SimulationSettings(sim, ::Type{FT}) where {FT <: AbstractFloat}
    SM = sampling_mode(sim)
    return SimulationSettings{SM,FT,typeof(sim.integrator)}(
        sim.integrator,
        FT(sim.Δt),
        FT(sim.t_final),
        FT(sim.noise_scale),
        FT(sim.μ),
        sim.M,
        FT.(sim.saveat),
    )
end

#####################################

"""
    predict(tnn, sim, inputs; seed=nothing)

Return the mean network output: at `sim.t_final` across `sim.M` independent
trajectories in reset mode, or across `sim.saveat` times on one continuous
trajectory per input in serial mode. Outputs are raw scores (logits for a
classifier); use `class_probabilities` to apply softmax after averaging.
Input columns represent samples; a vector represents one sample. The returned
matrix has shape `(n_outputs, n_points)`. Each trajectory starts from zero.
"""
function predict(
    tnn::TNN{FT},
    sim::SimulationSettings,
    inputs::AbstractVecOrMat{FT};
    seed=nothing,
) where {FT}
    size(inputs, 1) == n_inputs(params(tnn)) ||
        throw(DimensionMismatch("inputs must have $(n_inputs(params(tnn))) rows"))
    size(inputs, 2) > 0 || throw(ArgumentError("inputs must contain at least one point"))
    I = inputs isa AbstractVector ? reshape(inputs, length(inputs), 1) : inputs
    initial = similar(I, n_neurons(tnn), size(I, 2))
    fill!(initial, zero(FT))
    return _predict!(tnn, SimulationSettings(sim, FT), initial, I; seed)
end

function _predict!(
    tnn::TNN{FT},
    sim::SimulationSettings{SM,FT},
    X₀::T,
    I::T;
    seed = nothing,
    trajectory_seeds = nothing,
    noise_observer = nothing,
    ensemblealg = SDE.EnsembleSerial(),
) where {SM, FT, T <: AbstractVecOrMat{FT}}

    @assert size(I, 2) == size(X₀, 2) "Input and initial state batch size (2nd dim) must match got $(size(I,2)), $(size(X₀, 2))"

    function forcing!(du, u, parameters, t)
        X = reshape(u, size(X₀))
        F = reshape(du, size(X₀))
        fill!(du, zero(FT))
        p = _parameter_views(tnn, parameters)
        _add_F_internal!(F, neurons(tnn), p, X)
        _add_F_external!(F, p, I)
        du .*= sim.μ
        return nothing
    end

    function noise!(du, X, tnn, t)
        du .= sim.noise_scale
        return nothing
    end

    dynamics = SciMLBase.SDEFunction{true,SciMLBase.FullSpecialize}(forcing!, noise!)
    brownian = if trajectory_seeds === nothing
        SDE.WienerProcess!(zero(FT), zero(vec(X₀)))
    else
        # The solver reseeds this private noise RNG from the trajectory RNG.
        # Neither construction nor replay depends on the task-local RNG.
        SDE.WienerProcess!(zero(FT), zero(vec(X₀)); rng=Random.Xoshiro(0))
    end
    prob = SDE.SDEProblem(
        dynamics, noise!, vec(X₀), (zero(FT), sim.t_final), _pack_parameters(tnn);
        noise=brownian,
    )
    sol = _simulate!(prob, sim; seed, trajectory_seeds, ensemblealg)

    if noise_observer !== nothing
        noise_observer(sol)
    end

    @assert length(sol.u) == _trajectory_count(sim)
    @assert length(sol.u[1].u[end]) == length(X₀)

    N_last = n_last_layer_neurons(tnn)
    if SM === SerialMode
        trajectory = only(sol.u)
        X_mean = zeros(FT, N_last, size(I, 2))
        for t in sim.saveat
            idx = findfirst(==(t), trajectory.t)
            idx === nothing && error("serial observation time $t was not saved")
            X_mean .+= @view reshape(trajectory.u[idx], n_neurons(tnn), size(I, 2))[end-N_last+1:end, :]
        end
        X_mean ./= sim.M
        return output_coupling(tnn) * X_mean
    end
    final_states = (
        @view(reshape(trajectory.u[end], n_neurons(tnn), size(I, 2))[end-N_last+1:end, :])
        for trajectory in sol.u
    )

    X_mean = reduce(+, final_states) ./ length(sol.u)
    #! also get final layer var?

    @assert size(X_mean) == (N_last, size(I, 2)) # (n_neurons_final_layer, n_points)

    # Eqn 9 generalized for multi-output
    return output_coupling(tnn) * X_mean # (n_outputs, n_points)

end

_trajectory_count(sim::SimulationSettings{ResetMode}) = sim.M
_trajectory_count(::SimulationSettings{SerialMode}) = 1


function _trajectory_rng(ctx)
    ctx.sim_seed === nothing || Random.seed!(ctx.sim_seed)
    return Random.default_rng()
end

function _replay_rng(ctx, seeds)
    return Random.Xoshiro(seeds[ctx.sim_id])
end

function _simulate!(
    prob,
    sim;
    seed = nothing,
    trajectory_seeds = nothing,
    ensemblealg = SDE.EnsembleSerial(),
)
    trajectory_seeds === nothing || length(trajectory_seeds) == _trajectory_count(sim) ||
        throw(DimensionMismatch("one explicit seed is required per trajectory"))
    rng_factory = trajectory_seeds === nothing ? _trajectory_rng :
        ctx -> _replay_rng(ctx, trajectory_seeds)

    # Each threaded trajectory must own its mutable noise process and RNG.
    ensembleprob = SDE.EnsembleProblem(prob; safetycopy=ensemblealg isa SDE.EnsembleThreads)
    seed_kwargs = seed === nothing ? (;) : (; seed)
    sampling_kwargs = if sampling_mode(sim) === SerialMode
        (; saveat=sim.saveat, tstops=sim.saveat,
           save_start=first(sim.saveat) == 0, save_end=last(sim.saveat) == sim.t_final,
           save_everystep=false)
    else
        (; saveat=sim.t_final)
    end
    sol = SciMLBase.solve(
        ensembleprob,
        sim.integrator,
        ensemblealg;
        trajectories = _trajectory_count(sim),
        rng_func = rng_factory,
        dt=sim.Δt,
        adaptive=false, #! change?
        save_noise = true,
        progress = true,
        sampling_kwargs...,
        seed_kwargs...
    )

    for (i, trajectory) in enumerate(sol.u)
        if !SciMLBase.successful_retcode(trajectory)
            error("SDE trajectory $i failed with retcode=$(trajectory.retcode)")
        end
    end

   return sol

end
