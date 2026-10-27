export ResetMode, SerialMode, SimulationSettings

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
end

sampling_mode(::SimulationSettings{SM}) where SM = SM

function SimulationSettings(Δt, t_final, kBT, μ, M, ::Type{SM}; integrator = SDE.EM()) where {SM <: SamplingMode}

    # handle mixed float types
    FT = float(promote_type(
        typeof(Δt),
        typeof(t_final),
        typeof(kBT),
        typeof(μ),
    ))

    # Δt factor should come through integrator
    noise_scale = sqrt(2 * kBT * μ)

    return SimulationSettings{SM,FT,typeof(integrator)}(
        integrator,
        FT(Δt),
        FT(t_final),
        FT(noise_scale),
        FT(μ),
        M,
    )
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
    )
end

#####################################

function _predict!(
    tnn::TNN{FT},
    sim::SimulationSettings{ResetMode,FT},
    X₀::T,
    I::T,
    sensealg;
    seed = nothing,
) where {FT, T <: AbstractVecOrMat{FT}}

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
    # Brownian increments are independent of the parameters. Explicit Float storage
    # avoids ReverseDiff.TrackedArray noise buffers, which disallow mutation.
    brownian = Zygote.ignore() do
        SDE.WienerProcess!(zero(FT), zero(vec(X₀)))
    end
    prob = SDE.SDEProblem(
        dynamics, noise!, vec(X₀), (zero(FT), sim.t_final), _pack_parameters(tnn);
        noise=brownian,
    )
    sol = _simulate!(prob, sim, sensealg; seed)

    @assert length(sol.u) == sim.M "length(sol.u) != M you coded something wrong ethan"
    @assert length(sol.u[1].u[end]) == length(X₀)

    N_last = n_last_layer_neurons(tnn)
    # Could do views here
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


# Random seeds are independent of the model parameters. Exclude task-local RNG
# setup from AD (Julia 1.13's seed! accesses task-local storage).
function _trajectory_rng(ctx)
    return Zygote.ignore() do
        ctx.sim_seed === nothing || Random.seed!(ctx.sim_seed)
        Random.default_rng()
    end
end

function _simulate!(
    prob,
    sim,
    sensealg;
    seed = nothing,
)
    
    ensembleprob = SDE.EnsembleProblem(prob)
    seed_kwargs = seed === nothing ? (;) : (; seed)
    sol = SciMLBase.solve(
        ensembleprob,
        sim.integrator,
        # Threaded ReverseDiff ensembles failed the multi-trajectory FD check.
        SDE.EnsembleSerial();
        sensealg=sensealg,
        trajectories = sim.M,
        rng_func = _trajectory_rng,
        dt=sim.Δt,
        adaptive=false, #! change?
        save_noise = true,
        saveat = sim.t_final,
        progress = true,
        seed_kwargs...
    ) #! TODO SET THREADS OR SOMETHING: SDE.EnsembleThreads, EnsembleDistributed, EnsembleGPUArray

    for (i, trajectory) in enumerate(sol.u)
        if !SciMLBase.successful_retcode(trajectory)
            error("SDE trajectory $i failed with retcode=$(trajectory.retcode)")
        end
    end

   return sol

end
