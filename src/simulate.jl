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
    return SimulationSettings{SM,FT}(
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
    sensealg
) where {FT, T <: AbstractVecOrMat{FT}}

    @assert size(I, 2) == size(X₀, 2) "Input and initial state batch size (2nd dim) must match got $(size(I,2)), $(size(X₀, 2))"

    function forcing!(du, X, tnn, t)
        fill!(du, zero(FT))
        force!(du, tnn, X, I) 
        du .*= sim.μ
        return nothing
    end

    function noise!(du, X, tnn, t)
        du .= sim.noise_scale
        return nothing
    end

    prob = SDE.SDEProblem(forcing!, noise!, X₀, (0.0, sim.t_final), tnn)
    sol = _simulate!(prob, sim, sensealg)

    # I think sol.u is M length vector of Array(1, n_neurons, n_points)
    @assert length(sol.u) == sim.M "length(sol.u) != M you coded something wrong ethan"
    @assert size(sol.u[1].u[end]) == size(X₀)

    N_last = n_last_layer_neurons(tnn)
    # Could do views here
    final_states = (
        @view(trajectory.u[end][end-N_last+1:end, :])
        for trajectory in sol.u
    )

    X_mean = reduce(+, final_states) ./ length(sol.u)
    #! also get final layer var?

    @assert size(X_mean) == (N_last, size(I, 2)) # (n_neurons_final_layer, n_points)

    # Eqn 9 generalized for multi-output
    return output_coupling(tnn) * X_mean # (n_outputs, n_points)

end


function _simulate!(
    prob,
    sim,
    sensealg
)
    
    ensembleprob = SDE.EnsembleProblem(prob)
    sol = SciMLBase.solve(
        ensembleprob,
        sim.integrator;
        sensealg=sensealg,
        trajectories = sim.M,
        dt=sim.Δt,
        adaptive=false, #! change?
        saveat = sim.t_final,
        progress = true
    ) #! TODO SET THREADS OR SOMETHING: SDE.EnsembleThreads, EnsembleDistributed, EnsembleGPUArray

    for (i, trajectory) in enumerate(sol.u)
        if !SciMLBase.successful_retcode(trajectory)
            error("SDE trajectory $i failed with retcode=$(trajectory.retcode)")
        end
    end

   return sol

end
