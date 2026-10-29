using AnharmonicThermodynamicComputer, Random, Test
const ATC = AnharmonicThermodynamicComputer

@testset "Threaded replay" begin
    Random.seed!(74)
    x = [0.2 0.5 0.8; -0.1 0.3 0.2]
    y = [1.0 0 0; 0 1 0; 0 0 1]
    model = TNN(2, 2, 2, 3, x)
    sim = SimulationSettings(0.007, 0.035, 0.1, 0.7, 23, ResetMode)
    if Threads.nthreads() == 1
        @test_throws ArgumentError ThreadedReplayGradientCache(model, sim, x)
        @test_throws ArgumentError replay_loss_gradient(model, sim, x, y, ReplayPlan(sim; seed=1); threaded=true)
    else
        initial = 0.1 .* randn(4, 3)
        serial_cache = ReplayGradientCache(model, sim, x)
        for (batch_size, workers) in ((1, Threads.nthreads()), (10, 2), (10, Threads.nthreads()))
            pool = ThreadedReplayGradientCache(model, sim, x; batch_size, workers)
            @test length(pool.workers) == min(workers, batch_size, sim.M)
            @test pool.batch_size == length(pool.workers)
            @test size(pool.gradients, 2) == pool.batch_size
            for M in (1, 3, 23), loss in (mse_loss, categorical_crossentropy)
                settings = SimulationSettings(0.007, 0.035, 0.1, 0.7, M, ResetMode)
                plan = ReplayPlan(settings; seed=739)
                expected = replay_loss_gradient(model, settings, x, y, plan; loss, cache=serial_cache, initial_state=initial)
                for repeat in 1:2
                    Random.seed!(99)
                    next_draws = rand(4)
                    Random.seed!(99)
                    actual = replay_loss_gradient(model, settings, x, y, plan; loss, cache=pool, initial_state=initial)
                    @test rand(4) == next_draws
                    @test actual.predictions == expected.predictions
                    @test actual.loss == expected.loss
                    @test actual.noise_fingerprints == expected.noise_fingerprints
                    @test actual.seeds == expected.seeds
                    for name in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
                        @test getproperty(actual.gradient.params, name) ≈ getproperty(expected.gradient.params, name) rtol=1e-12 atol=1e-14
                    end
                end
            end
            # Reuse with new parameters, inputs, targets, initial state, and seed.
            changed = deepcopy(model)
            changed.params.bᵢ .+= 0.03
            changed.params.fᵢ .*= 1.1
            plan = ReplayPlan(sim; seed=920)
            expected = replay_loss_gradient(changed, sim, x .* 0.8, y[:, [3, 2, 1]], plan;
                loss=categorical_crossentropy, cache=serial_cache, initial_state=initial .+ 0.02)
            actual = replay_loss_gradient(changed, sim, x .* 0.8, y[:, [3, 2, 1]], plan;
                loss=categorical_crossentropy, cache=pool, initial_state=initial .+ 0.02)
            @test actual.predictions == expected.predictions
            saved = deepcopy(actual)
            unchecked = replay_loss_gradient(changed, sim, x .* 0.8, y[:, [3, 2, 1]], plan;
                loss=categorical_crossentropy, cache=pool, initial_state=initial .+ 0.02, check_replay=false)
            @test unchecked.noise_fingerprints === nothing
            @test actual == saved
            for name in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
                @test getproperty(actual.gradient.params, name) ≈ getproperty(expected.gradient.params, name) rtol=1e-12 atol=1e-14
                @test getproperty(unchecked.gradient.params, name) == getproperty(actual.gradient.params, name)
            end
            wrong = SimulationSettings(0.01, 0.035, 0.1, 0.7, 23, ResetMode)
            @test_throws ArgumentError replay_mse_gradient(model, wrong, x, y, plan; cache=pool)
            @test_throws DimensionMismatch replay_mse_gradient(model, sim, x[:, 1:2], y[:, 1:2], plan; cache=pool)
        end
        plan = ReplayPlan(sim; seed=739)
        automatic = replay_mse_gradient(model, sim, x, y, plan; threaded=true, trajectory_batch_size=10)
        expected = replay_mse_gradient(model, sim, x, y, plan; cache=serial_cache)
        @test automatic.predictions == expected.predictions
        @test automatic.gradient.params.Jᵢⱼ ≈ expected.gradient.params.Jᵢⱼ
        @test_throws ArgumentError replay_mse_gradient(model, sim, x, y, plan; threaded=true, cache=serial_cache)
        @test_throws ArgumentError ThreadedReplayGradientCache(model, sim, x; batch_size=0)
        @test_throws ArgumentError ThreadedReplayGradientCache(model, sim, x; workers=0)
        serial_mode = SimulationSettings(0.01, 0.03, 0.1, 1.0, SerialMode; saveat=[0.01, 0.03])
        @test_throws ArgumentError ThreadedReplayGradientCache(model, serial_mode, x)

        xf, yf = Float32.(x), Float32.(y)
        mf = TNN(2, 2, 2, 3, xf)
        sf = SimulationSettings(0.01, 0.03, 0.1, 0.7, 3, ResetMode)
        pf = ReplayPlan(sf; seed=1)
        af = replay_loss_gradient(mf, sf, xf, yf, pf; threaded=true, loss=categorical_crossentropy)
        ef = replay_loss_gradient(mf, sf, xf, yf, pf; loss=categorical_crossentropy)
        @test af.predictions == ef.predictions
        @test af.gradient.params.Jᵢⱼ ≈ ef.gradient.params.Jᵢⱼ
    end
end

if Threads.nthreads() > 1
    @testset "Threaded training and bounded final snapshots" begin
        x = reshape([-0.5, 0.5], 1, :)
        y = [1.0 0.0; 0.0 1.0]
        sim = SimulationSettings(0.01, 0.03, 0.1, 1.0, 13, ResetMode)
        for loss in (mse_loss, categorical_crossentropy), nthreads in unique([2, Threads.nthreads()])
            Random.seed!(1234)
            expected, expected_losses, expected_saved = train(1, 1, 2, 2, sim, x, y;
                loss, maxiters=2, saveat=[0, 1], nthreads=1)
            next_draw = rand(UInt64)
            Random.seed!(1234)
            actual, losses, saved = train(1, 1, 2, 2, sim, x, y;
                loss, maxiters=2, saveat=[0, 1], nthreads, trajectory_batch_size=10)
            @test rand(UInt64) == next_draw
            @test losses ≈ expected_losses
            @test saved.predictions ≈ expected_saved.predictions
            for name in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
                @test getproperty(actual.params, name) ≈ getproperty(expected.params, name)
            end
        end
        # Final snapshots use bounded forward-only ensembles, without a new AD tape.
        noiseless = SimulationSettings(0.01, 0.03, 0.0, 1.0, 13, ResetMode)
        for iterations in (0, 2)
            model, losses, saved = train(1, 1, 2, 2, noiseless, x, y;
                maxiters=iterations, saveat=[iterations])
            @test only(saved.predictions) ≈ predict(model, noiseless, x)
            @test length(losses) == iterations
        end
    end
end

@testset "Training thread count validation and defaults" begin
    x = reshape([-0.5, 0.5], 1, :)
    y = copy(x)
    reset = SimulationSettings(0.01, 0.03, 0.1, 1.0, 3, ResetMode)
    serial = SimulationSettings(0.01, 0.03, 0.1, 1.0, SerialMode; saveat=[0.01, 0.03])
    for sim in (reset, serial)
        for invalid in (0, -1, Threads.nthreads() + 1)
            @test_throws ArgumentError train(1, 1, 2, 1, sim, x, y; nthreads=invalid, maxiters=0)
        end
        @test_throws ArgumentError train(1, 1, 2, 1, sim, x, y; trajectory_batch_size=0, maxiters=0)
        Random.seed!(431)
        expected, losses, saved = train(1, 1, 2, 1, sim, x, y; nthreads=1, maxiters=1, saveat=[0])
        Random.seed!(431)
        actual, default_losses, default_saved = train(1, 1, 2, 1, sim, x, y; maxiters=1, saveat=[0])
        @test default_losses ≈ losses
        @test default_saved.predictions ≈ saved.predictions
        @test actual.params.Jᵢⱼ ≈ expected.params.Jᵢⱼ
    end
end
