export MSE

#### LOSS FUNCTIONS ####

function MSE(y_pred, targets)
    yp_sz = size(y_pred) 
    t_sz = size(targets)
    # Both should be (n_outputs, n_points)
    @assert yp_sz == t_sz "Prediction $(yp_sz) and target $(t_sz) shapes must match for MSE loss"

    return sum(abs2, y_pred .- targets) / yp_sz[2]
end

#########################

struct GeneticOptimizer{L, T}
    n_candidates::Int
    n_kept::Int # The best `n_kept` models are retained each iteration
    loss_fn::L
    mutation_scale::T
end

function mutate_parameters!(tnn::AllToAllTNN{FT}, σ::Real) where FT
    σ = FT(σ)

    b = bias(tnn)
    W = external_coupling(tnn)
    J = internal_coupling(tnn)
    f = tnn.fᵢ

    #! kinda expensive to duplicate like this
    Δb = similar(b)
    ΔW = similar(W)
    ΔJ = similar(J)
    Δf = similar(f)

    randn!(Δb)
    randn!(ΔW)
    randn!(ΔJ)
    randn!(Δf)

    b .+= σ .* Δb
    f .+= σ .* Δf
    W .+= σ .* ΔW .* tnn.W_scale

    J .+= (σ / sqrt(FT(2))) .*
          (ΔJ .+ transpose(ΔJ)) .*
          tnn.J_scale

    return tnn
end


function run_sim_factory(factory::Function, tnn::AllToAllTNN)
    sim = factory(tnn)
    sim isa Simulation || throw(ArgumentError(
        "simulation_factory(tnn) returned $(typeof(candidate)). Must be <: Simulation"
    ))
    return sim
end

run_sim_factory(::Function, ::T) where T = error("tnn_factory() returned type $(T), expected AllToAllTNN")


#! look into Evolutionary.jl
function train_genetic(
    go::GeneticOptimizer,
    tnn_factory,
    simulation_factory,
    z::AbstractVecOrMat,
    targets::AbstractVecOrMat;
    n_generations::Int,
)
    @assert 1 <= go.n_kept <= go.n_candidates
    @assert n_generations >= 1

    candidates = [tnn_factory() for _ in 1:go.n_candidates]
    offspring = similar.(candidates)

    tnn = first(candidates)
    @assert size(z, 1) == size(external_coupling(tnn), 1) "Input feature count must match W"

    # Reused across sequential evaluations of identically shaped networks
    sim = simulation_factory(tnn)

    p = Progress(n_generations; dt = 1.0) # would be nice to have this also tick up with each M that finishes
    for generation in 1:n_generations
        # Evaluate current population
        losses = map(candidates) do candidate
            y_pred = _estimate!(candidate, sim, z)
            go.loss_fn(y_pred, targets)
        end

        # Rank candidates by increasing loss
        ranking = sortperm(losses)
        selected = ranking[1:go.n_kept]

        # Can finish early
        if generation == n_generations
            return candidates[first(ranking)]
        end

        # Cycle through selected parents to populate all offspring buffers
        # Maps n_candidates // n_kept
        for (child, parent_idx) in zip(offspring, Iterators.cycle(selected))
            copy_parameters!(child, candidates[parent_idx])
            mutate_parameters!(child, go.mutation_scale)
        end

        candidates, offspring = offspring, candidates
        next!(p)
    end
    finish!(p)
end


