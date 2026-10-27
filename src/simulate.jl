export ResetMode, SerialMode, SimulationSettings

abstract type SamplingMode end

struct ResetMode <: SamplingMode end
struct SerialMode <: SamplingMode end

struct SimulationSettings{
    FT <: AbstractFloat,
    SM <: SamplingMode
}
    Δt::FT
    t_final::FT
    noise_scale::FT
    μ::FT
    M::Int
end

function SimulationSettings(Δt, t_final, kBT, μ, M, ::Type{SM}) where {FT, SM <: SamplingMode}

    @assert size(I, 1) == size(external_coupling(tnn), 1) "Input feature count must match the number of rows in W"

    noise_scale = sqrt(2 * kBT * μ * Δt)

    return SimulationSettings{FT,SM}(FT(Δt), FT(t_final), FT(noise_scale), FT(μ), M)
end

function _predict!(
    tnn::AllToAllTNN{FT},
    sim::SimulationSettings{FT},
    X₀::T,
    I::T,
    sensalg
) where {FT, T <: AbstractVecOrMat{FT}}

    @assert size(I, 2) == size(X₀, 2) "Input and initial state batch size (2nd dim) must match got $(size(I,2)), $(size(X₀, 2))"

    function forcing!(du, X, p, t)
        force!(du, tnn, X, I) 
        du .*= p.μ
    end

    #! ok to capture noise scale for AD?
    noise!(du, X, p, t) = (du .= sim.noise_scale)

    #! double check sim.X gets copied here (X₀) not aliased
    X₀ .= FT(0.0)
    prob = SDEProblem(forcing!, noise!, X₀, (0.0, sim.t_final), params(tnn))
    sol = _simulate!(prob, sim, I)

    # I think sol.u is M length vector of Array(1, n_neurons, n_points)
    @assert length(sol.u == M) "length(sol.u) != M you coded something wrong ethan"
    @assert size(sol.u[1]) == (1, length(tnn), size(I, 2)) # check size of a trajectory

    N_last = neurons_in_last_layer(tnn)
    final_states = (@view(x_i.u[1][end - N_last + 1:end, :]) for x_i in sol.u)
    X_mean = reduce(+, final_states) ./ length(sol.u)
    #! also get final layer var?

    @assert size(X_mean) == (N_last, size(I, 2)) # (n_neurons_final_layer, n_points)

    # Eqn 9 generalized for multi-output
    return tnn.fᵢ * X_mean # (n_outputs, n_points)

end


function _simulate!(
    prob,
    sim_settings,
    I::AbstractVecOrMat{FT},
    sensalg
) where FT
    
    ensembleprob = SDE.EnsembleProblem(prob)
    sol = solve(
        ensembleprob,
        sim.solver;
        sensalg=sensalg,
        trajectories = sim.M,
        dt=sim.Δt,
        adaptive= false, #! change?
        saveat = sim.t_final,
        progress = true
    ) #! TODO SET THREADS OR SOMETHING: SDE.EnsembleThreads, EnsembleDistributed, EnsembleGPUArray

    failure = !successful_retcode(sol)
    failure && error("Solver failed retcode=$(sol.retcode)")

   return sol

end
