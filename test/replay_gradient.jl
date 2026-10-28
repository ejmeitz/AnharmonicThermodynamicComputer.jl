using AnharmonicThermodynamicComputer, ReverseDiff, Optimisers
using Test, Random

const ATC = AnharmonicThermodynamicComputer
const PARAMETER_FIELDS = (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)

function noise_history(sol)
    [(times=copy(s.W.t), values=deepcopy(s.W.W)) for s in sol.u]
end

@testset "Two-pass trajectory replay" begin
    Random.seed!(123)
    inputs = randn(2, 3)
    targets = randn(2, 3)
    initial = 0.1 .* randn(4, 3)
    model = TNN(2, 2, 2, 2, inputs)
    original_model = deepcopy(model)
    original_initial = copy(initial)

    for kBT in (0.0, 0.1), M in (1, 3), dt in (0.01, 0.005)
        @testset "kBT=$kBT M=$M dt=$dt" begin
            println("Checking replay: kBT=$kBT M=$M dt=$dt")
            flush(stdout)
            sim = SimulationSettings(dt, 0.05, kBT, 0.7, M, ResetMode)
            plan = ReplayPlan(M; seed=739)
            forward_noise = Ref{Any}()
            repeated_noise = Ref{Any}()
            forward_observer = sol -> (forward_noise[] = noise_history(sol))
            repeated_observer = sol -> (repeated_noise[] = noise_history(sol))
            reference_prediction = ATC._predict!(model, sim, initial, inputs;
                seed=0, trajectory_seeds=plan.seeds, noise_observer=forward_observer)
            # Consume unrelated randomness between repeated forward solves.
            randn(123)
            reference_loss(m) = sum(abs2, ATC._predict!(m, sim, initial, inputs;
                seed=0, trajectory_seeds=plan.seeds, noise_observer=repeated_observer) .- targets) / size(inputs, 2)
            reference_value = reference_loss(model)
            @test isequal(forward_noise[], repeated_noise[])
            @test all(h -> length(h.times) > 2, forward_noise[])
            if M > 1
                @test !isequal(forward_noise[][1].values, forward_noise[][2].values)
            end

            # Compare each complete path to the ensemble, then feed those exact
            # increments to the reusable tape and check its actual prediction.
            one_sim = ATC._single_trajectory_settings(sim)
            cache = ReplayGradientCache(model, sim, inputs)
            cache.data.tape_inputs[4] .= initial
            cache.data.tape_inputs[5] .= 1
            for i in 1:M
                single_forward = Ref{Any}()
                single_prediction = ATC._predict!(model, one_sim, initial, inputs;
                    seed=0, trajectory_seeds=plan.seeds[i:i],
                    noise_observer=sol -> begin
                        single_forward[] = noise_history(sol)
                        ATC._copy_noise_increments!(cache.data.tape_inputs[2], cache.data.times, sol)
                    end)
                @test isequal(only(single_forward[]), forward_noise[][i])
                ReverseDiff.gradient!(cache.data.derivatives, cache.tape, cache.data.tape_inputs)
                @test last(cache.data.derivatives) ≈ single_prediction rtol=1e-12 atol=1e-14
            end

            cached_result = replay_mse_gradient(model, sim, inputs, targets, plan; initial_state=initial, cache)
            result = replay_mse_gradient(model, sim, inputs, targets, plan; initial_state=initial)
            @test result.loss ≈ reference_value rtol=1e-13 atol=1e-14
            @test result.predictions ≈ reference_prediction rtol=1e-13 atol=1e-14
            @test result.seeds == plan.seeds
            @test length(result.noise_fingerprints) == M
            max_error = 0.0
            for field in PARAMETER_FIELDS
                actual = getproperty(result.gradient.params, field)
                expected = getproperty(cached_result.gradient.params, field)
                @test actual ≈ expected rtol=1e-10 atol=1e-12
                max_error = max(max_error, maximum(abs, actual .- expected))
            end
            println("Maximum retained-cache–fresh-cache gradient difference: $max_error")

            if kBT == 0.1 && M == 3 && dt == 0.01
                # Independent finite differences, with every parameter entry perturbed.
                fd_loss(m) = sum(abs2, ATC._predict!(m, sim, initial, inputs;
                    seed=0, trajectory_seeds=plan.seeds) .- targets) / size(inputs, 2)
                max_fd_error = 0.0
                for field in PARAMETER_FIELDS
                    values = getproperty(model.params, field)
                    grad = getproperty(result.gradient.params, field)
                    for i in eachindex(values)
                        original = values[i]
                        h = 1e-5
                        fd = try
                            values[i] = original + h
                            plus = fd_loss(model)
                            values[i] = original - h
                            minus = fd_loss(model)
                            (plus - minus) / (2h)
                        finally
                            values[i] = original
                        end
                        max_fd_error = max(max_fd_error, abs(grad[i] - fd))
                        @test isapprox(grad[i], fd; rtol=1e-5, atol=1e-8)
                    end
                end
                println("Maximum replay–FD gradient error: $max_fd_error")

                reordered = replay_mse_gradient(model, sim, inputs, targets,
                    ReplayPlan(reverse(plan.seeds)); initial_state=initial, check_replay=false)
                @test reordered.noise_fingerprints === nothing
                @test reordered.predictions ≈ result.predictions
                for field in PARAMETER_FIELDS
                    @test getproperty(reordered.gradient.params, field) ≈ getproperty(result.gradient.params, field)
                end

                # Check compatibility with the optimizer, not only array values.
                replay_model, reference_model = deepcopy(model), deepcopy(model)
                opt1, opt2 = Optimisers.setup(Optimisers.Adam(), replay_model), Optimisers.setup(Optimisers.Adam(), reference_model)
                _, replay_model = Optimisers.update!(opt1, replay_model, result.gradient)
                _, reference_model = Optimisers.update!(opt2, reference_model, cached_result.gradient)
                for field in PARAMETER_FIELDS
                    @test getproperty(replay_model.params, field) ≈ getproperty(reference_model.params, field)
                end

                # Neither plan construction nor replay should consume the caller's RNG.
                Random.seed!(845)
                expected_draws = rand(5)
                Random.seed!(845)
                ReplayPlan(M; seed=22)
                repeated = replay_mse_gradient(model, sim, inputs, targets, plan; initial_state=initial)
                @test rand(5) == expected_draws
                @test repeated.predictions == result.predictions
                @test repeated.noise_fingerprints == result.noise_fingerprints
                for field in PARAMETER_FIELDS
                    @test getproperty(repeated.gradient.params, field) == getproperty(result.gradient.params, field)
                end
                different = replay_mse_gradient(model, sim, inputs, targets, ReplayPlan(M; seed=740); initial_state=initial)
                @test different.noise_fingerprints != result.noise_fingerprints
                @test different.predictions != result.predictions
            end
        end
    end
    for field in PARAMETER_FIELDS
        @test getproperty(model.params, field) == getproperty(original_model.params, field)
    end
    @test initial == original_initial
    sim = SimulationSettings(0.01, 0.05, 0.1, 0.7, 3, ResetMode)
    @test_throws DimensionMismatch replay_mse_gradient(model, sim, inputs, targets, ReplayPlan(1; seed=1))
    @test_throws ArgumentError ReplayPlan(0; seed=1)
end
