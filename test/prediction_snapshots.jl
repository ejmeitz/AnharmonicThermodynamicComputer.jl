using AnharmonicThermodynamicComputer
using Random
using Test
using Optimisers

@testset "Public prediction and training snapshots" begin
    inputs = reshape([0.0, 0.5, 1.0], 1, :)
    targets = cospi.(2 .* inputs)
    sim = SimulationSettings(0.01, 0.02, 0.0, 1.0, 2, ResetMode)

    # A single harmonic neuron has an exact Euler recurrence:
    # x[n+1] = (1 - 2dt)x[n] + dt*z, with x[0] = 0.
    harmonic = TNN(1, 1, 1, 1, inputs; J₄=0.0)
    harmonic.params.bᵢ .= 0
    harmonic.params.Jᵢⱼ .= 0
    harmonic.params.Wᵢⱼ .= -1
    harmonic.params.fᵢ .= 1
    expected = inputs .* (1 - (1 - 2sim.Δt)^2) ./ 2
    @test predict(harmonic, sim, inputs; seed=7) ≈ expected
    @test predict(harmonic, sim, [0.5]; seed=7) ≈ expected[:, 2:2]
    @test_throws DimensionMismatch predict(harmonic, sim, zeros(2, 3))
    @test_throws ArgumentError predict(harmonic, sim, zeros(1, 0))

    Random.seed!(123)
    model, losses, saved = train(1, 1, 2, 1, sim, inputs, targets;
                                 maxiters=3, saveat=[0, 1, 3])
    @test saved.iterations == [0, 1, 3]
    @test length(saved.predictions) == 3
    @test all(p -> size(p) == size(targets), saved.predictions)
    @test all(isfinite, losses)
    @test saved.predictions[end] ≈ predict(model, sim, inputs)

    # A snapshot at k must match a model trained for exactly k updates.
    # This checks the initial snapshot, an intermediate one, and the final one.
    for (k, prediction) in zip(saved.iterations, saved.predictions)
        Random.seed!(123)
        reference, reference_losses = train(1, 1, 2, 1, sim, inputs, targets; maxiters=k)
        @test prediction ≈ predict(reference, sim, inputs)
        @test reference_losses ≈ losses[1:k]
        if k < length(losses)
            @test sum(abs2, prediction .- targets) / size(inputs, 2) ≈ losses[k + 1]
        end
    end
    @test saved.predictions[1] !== saved.predictions[2]

    # Saving intermediate curves must not perturb stochastic training or its RNG.
    noisy = SimulationSettings(0.01, 0.02, 0.1, 1.0, 2, ResetMode)
    Random.seed!(456)
    plain, plain_losses = train(1, 1, 2, 1, noisy, inputs, targets; maxiters=2)
    Random.seed!(456)
    recorded, recorded_losses, snapshots = train(1, 1, 2, 1, noisy, inputs, targets;
                                                 maxiters=2, saveat=[0, 1])
    @test plain_losses ≈ recorded_losses
    @test predict(plain, noisy, inputs; seed=8) ≈ predict(recorded, noisy, inputs; seed=8)
    for (k, prediction) in zip(snapshots.iterations, snapshots.predictions)
        @test sum(abs2, prediction .- targets) / size(inputs, 2) ≈ recorded_losses[k + 1]
    end

    # Zero-update and empty schedules have well-defined return shapes.
    untrained, empty_losses, initial = train(1, 1, 2, 1, sim, inputs, targets;
                                            maxiters=0, saveat=[0])
    @test isempty(empty_losses)
    @test only(initial.predictions) ≈ predict(untrained, sim, inputs)
    _, _, empty_saved = train(1, 1, 2, 1, sim, inputs, targets; maxiters=0, saveat=Int[])
    @test isempty(empty_saved.iterations) && isempty(empty_saved.predictions)
    for invalid in ([-1], [4], [1, 0], [1, 1], [0.5])
        @test_throws ArgumentError train(1, 1, 2, 1, sim, inputs, targets;
                                         maxiters=3, saveat=invalid)
    end
end

@testset "Training uses one reusable tape with fresh plans" begin
    inputs = reshape([0.0, 0.5, 1.0], 1, :)
    targets = cospi.(2 .* inputs)
    sim = SimulationSettings(0.01, 0.03, 0.1, 0.7, 3, ResetMode)
    Random.seed!(739)
    expected = TNN(1, 2, 2, 1, inputs)
    cache = ReplayGradientCache(expected, sim, inputs)
    state = Optimisers.setup(Optimisers.Adam(0.01), expected)
    losses = Float64[]
    curves = Matrix{Float64}[]
    plans = ReplayPlan[]
    for _ in 1:2
        plan = ReplayPlan(sim.M; seed=rand(UInt64))
        push!(plans, plan)
        result = replay_mse_gradient(expected, sim, inputs, targets, plan; cache)
        push!(losses, result.loss)
        push!(curves, result.predictions)
        state, expected = Optimisers.update(state, expected, result.gradient)
        enforce_connectivity!(expected)
    end
    expected_next_random = rand(UInt64)
    @test plans[1].seeds != plans[2].seeds

    Random.seed!(739)
    actual, actual_losses, saved = train(1, 2, 2, 1, sim, inputs, targets;
        maxiters=2, saveat=[0, 1])
    @test actual_losses ≈ losses
    @test saved.predictions ≈ curves
    @test rand(UInt64) == expected_next_random
    for field in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
        @test getproperty(actual.params, field) ≈ getproperty(expected.params, field)
    end

    _, vector_losses, vector_saved = train(1, 1, 2, 1, sim, [0.5], [0.0];
        maxiters=1, saveat=[0, 1])
    @test length(vector_losses) == 1
    @test all(p -> size(p) == (1, 1), vector_saved.predictions)
end
