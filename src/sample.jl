function _estimate!(
    tnn::AllToAllTNN{FT},
    sim::Simulation{FT},
    I::AbstractVecOrMat{FT}, # If vec batch_size = 1, else treat each col as a batch
) where FT

    @assert size(I, 2) == size(sim.X, 2) "Input batch size must match the simulation passed to _estimate"

    reset_state!(sim)
    X_mean = _sample!(sim, tnn, I, sim.mode) # (n_neurons_final_layer, n_points)

    # Eqn 9 generalized for multi-output
    return tnn.fᵢ * X_mean # (n_outputs, n_points)

end

function _sample!(
    sim::Simulation{FT},
    tnn::AllToAllTNN{FT},
    I::AbstractVecOrMat{FT},
    ::Type{ResetMode}
) where FT

    N_steps = length(sim)
    N_last = neurons_in_last_layer(tnn)
    n_points = size(I, 2)
    final_layer_results = similar(I, N_last, n_points)
    final_layer_results .= zero(FT)

    sample_progress = Progress(sim.M; desc="Reset samples: ", offset=1)
    for m in 1:sim.M #! can parallelize, need independent storage for X, η, and output
        # Evolve system according to dynamics
        simulate!(sim, tnn, I, N_steps)
        # Sample final layer neurons and reset
        final_layer_results .+= @view sim.X[end - N_last + 1 : end, :]
        zero_state_vector!(sim)
        next!(sample_progress)
    end
    finish!(sample_progress)

    # return state averaged over all trajectories
    final_layer_results ./= FT(sim.M)
    return final_layer_results # (n_neurons_final_layer, n_points)

end

function _sample!(
    sim::Simulation{FT},
    tnn::AllToAllTNN{FT},
    I::AbstractVecOrMat{FT},
    ::Type{SerialMode}
) where FT
    error("Serial mode not implemented yet.")
end

function simulate!(
    sim::Simulation{FT},
    tnn::AllToAllTNN{FT},
    I::AbstractVecOrMat{FT},
    N_steps::Int
) where FT

    for _ in 1:N_steps
        step!(sim, tnn, I)
    end
    return sim
end

#TODO Try other time-stepping via DifferentialEquations.jl
function step!(
    sim::Simulation{FT},
    tnn::AllToAllTNN{FT},
    I::AbstractVecOrMat{FT}
) where FT

    zero_force_buffer!(sim)
    force!(sim.F_buffer, tnn, sim.X, I)
    randn!(sim.η_buffer)

    # Euler–Maruyama
    sim.X .+= (sim.drift_scale .* sim.F_buffer) .+ (sim.noise_scale .* sim.η_buffer)

    return sim
end