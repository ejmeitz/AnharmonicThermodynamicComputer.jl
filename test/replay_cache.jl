using AnharmonicThermodynamicComputer, Optimisers
using Test, Random

const ATC = AnharmonicThermodynamicComputer
const FIELDS = (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)

function reference_loss(model, sim, initial, inputs, targets, plan)
    prediction = ATC._predict!(model, sim, initial, inputs;
        seed=0, trajectory_seeds=plan.seeds)
    return sum(abs2, prediction .- targets) / size(inputs, 2)
end

function compare_reference(model, sim, initial, inputs, targets, plan, cache)
    actual = replay_mse_gradient(model, sim, inputs, targets, plan; initial_state=initial, cache)
    value = reference_loss(model, sim, initial, inputs, targets, plan)
    @test actual.loss ≈ value rtol=1e-12 atol=1e-14
    max_error = 0.0
    for field in FIELDS
        a = getproperty(actual.gradient.params, field)
        parameters = getproperty(model.params, field)
        for i in eachindex(parameters)
            original = parameters[i]
            h = 1e-5
            fd = try
                parameters[i] = original + h
                plus = reference_loss(model, sim, initial, inputs, targets, plan)
                parameters[i] = original - h
                minus = reference_loss(model, sim, initial, inputs, targets, plan)
                (plus - minus) / (2h)
            finally
                parameters[i] = original
            end
            @test isapprox(a[i], fd; rtol=1e-5, atol=1e-8)
            max_error = max(max_error, abs(a[i] - fd))
        end
    end
    println("Maximum cached–SDE finite-difference error: $max_error")
    flush(stdout)
    return actual
end

@testset "Reusable ReverseDiff tape" begin
    Random.seed!(123)
    inputs = randn(2, 3)
    targets = randn(2, 3)
    initial = 0.1 .* randn(4, 3)
    model = TNN(2, 2, 2, 2, inputs; J₃=0.2)
    initial_model = deepcopy(model)
    for kBT in (0.0, 0.1), dt in (0.01, 0.007)
        @testset "kBT=$kBT dt=$dt" begin
            println("Checking cached replay: kBT=$kBT dt=$dt")
            flush(stdout)
            # dt=.007 exercises a shortened final step.
            sim = SimulationSettings(dt, 0.05, kBT, 0.7, 3, ResetMode)
            plan = ReplayPlan(3; seed=739)
            cache = ReplayGradientCache(model, sim, inputs)
            result = compare_reference(model, sim, initial, inputs, targets, plan, cache)
            saved_result = deepcopy(result)
            replay = replay_mse_gradient(model, sim, inputs, targets, plan; initial_state=initial)
            @test result.predictions == replay.predictions
            @test result.noise_fingerprints == replay.noise_fingerprints

            # Different trajectory count, noise, parameters, targets, inputs, and
            # initial states must all flow through the SAME recorded tape.
            updated = deepcopy(model)
            for field in FIELDS
                getproperty(updated.params, field) .+= 0.01
            end
            sim4 = SimulationSettings(dt, 0.05, kBT, 0.7, 4, ResetMode)
            changed = compare_reference(updated, sim4, initial .+ 0.03, inputs .* 1.2,
                targets .- 0.4, ReplayPlan(4; seed=865), cache)
            @test changed.predictions != result.predictions
            @test result == saved_result

            # Revisit the first inputs after another call: no stale derivatives.
            Random.seed!(42)
            expected_draws = rand(5)
            Random.seed!(42)
            repeated = replay_mse_gradient(model, sim, inputs, targets, plan;
                initial_state=initial, cache)
            @test rand(5) == expected_draws
            @test repeated == result
            unchecked = replay_mse_gradient(model, sim, inputs, targets, plan;
                initial_state=initial, cache, check_replay=false)
            @test unchecked.gradient == result.gradient
            @test unchecked.noise_fingerprints === nothing

            # Verify the explicit noise buffer receives precisely the increments
            # reconstructed from the last reference Brownian history.
            history = Ref{Any}()
            ATC._predict!(model, ATC._single_trajectory_settings(sim), initial, inputs;
                seed=0, trajectory_seeds=plan.seeds[end:end],
                noise_observer=sol -> (history[] = only(sol.u).W))
            @test cache.data.times == history[].t
            for s in axes(cache.data.tape_inputs[2], 2)
                @test cache.data.tape_inputs[2][:, s] == history[].W[s + 1] .- history[].W[s]
            end

            if kBT == 0.1 && dt == 0.007
                trained = deepcopy(model)
                opt = Optimisers.setup(Optimisers.Adam(), trained)
                for iteration in 1:2
                    step = replay_mse_gradient(trained, sim, inputs, targets,
                        ReplayPlan(3; seed=900 + iteration); cache)
                    opt, trained = Optimisers.update!(opt, trained, step.gradient)
                    enforce_connectivity!(trained)
                end
                compare_reference(trained, sim, zero(initial), inputs, targets, plan, cache)
            end

            bad_sim = SimulationSettings(dt, 0.05, kBT, 0.8, 3, ResetMode)
            @test_throws ArgumentError replay_mse_gradient(model, bad_sim, inputs, targets, plan; cache)
            @test_throws DimensionMismatch replay_mse_gradient(model, sim, inputs[:, 1:2], targets[:, 1:2], plan; cache)
            changed_neurons = TNN(2, 2, 2, 2, inputs; J₄=0.5)
            @test_throws ArgumentError replay_mse_gradient(changed_neurons, sim, inputs, targets, plan; cache)
        end
    end
    for field in FIELDS
        @test getproperty(model.params, field) == getproperty(initial_model.params, field)
    end

    # Float32, one layer, and one step (dt greater than final time).
    x = Float32[0.2 0.7]
    m = TNN(1, 1, 2, 1, x; J₄=0f0)
    s = SimulationSettings(0.1f0, 0.05f0, 0.1f0, 0.7f0, 2, ResetMode)
    p = ReplayPlan(2; seed=123)
    c = ReplayGradientCache(m, s, x)
    r = replay_mse_gradient(m, s, x, cos.(x), p; cache=c)
    ref = replay_mse_gradient(m, s, x, cos.(x), p)
    @test r.predictions == ref.predictions
    for field in FIELDS
        @test getproperty(r.gradient.params, field) ≈ getproperty(ref.gradient.params, field) rtol=1e-4 atol=1e-6
    end
end
