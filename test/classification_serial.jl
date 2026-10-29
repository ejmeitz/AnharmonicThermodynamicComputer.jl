using AnharmonicThermodynamicComputer, Random, Test

const ATC = AnharmonicThermodynamicComputer

@testset "Stable categorical loss" begin
    y = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    @test categorical_crossentropy(zeros(3, 2), y) ≈ log(3)
    scores = [1000.0 -1000.0; -1000.0 1000.0; 0.0 0.0]
    @test categorical_crossentropy(scores, y) ≈ 0 atol=1e-12
    @test categorical_crossentropy(scores, y[[2, 1, 3], :]) ≈ 2000
    @test class_probabilities(scores) ≈ y
    @test all(isfinite, class_probabilities(scores))
    @test vec(sum(class_probabilities(scores); dims=1)) ≈ ones(2)
    scores = [0.3 -0.6; -0.5 0.8; 1.2 -0.1]
    @test categorical_crossentropy(scores .+ [10000.0 -10000.0], y) ≈ categorical_crossentropy(scores, y) atol=1e-11
    for targets in (y, [0.2 0.1; 0.3 0.7; 0.5 0.2])
        gradient = zero(scores)
        value = ATC._loss_and_cotangent!(gradient, categorical_crossentropy, scores, targets)
        @test value == categorical_crossentropy(scores, targets)
        for i in eachindex(scores)
            plus, minus = copy(scores), copy(scores)
            plus[i] += 1e-5
            minus[i] -= 1e-5
            fd = (categorical_crossentropy(plus, targets)-categorical_crossentropy(minus, targets))/2e-5
            @test gradient[i] ≈ fd rtol=1e-6 atol=1e-9
        end
    end
    @test_throws DimensionMismatch categorical_crossentropy(scores, zeros(2, 2))
    for bad in (zeros(3, 2), 2y, fill(NaN, 3, 2), [-0.1 0.0; 1.1 1.0; 0.0 0.0])
        @test_throws ArgumentError categorical_crossentropy(scores, bad)
    end
    @test_throws ArgumentError categorical_crossentropy(fill(Inf, 3, 2), y)
end

@testset "Serial means and settings" begin
    x = reshape([0.0, 0.5, 1.0], 1, :)
    model = TNN(1, 1, 1, 1, x; J₄=0.0)
    model.params.bᵢ .= 0
    model.params.Jᵢⱼ .= 0
    model.params.Wᵢⱼ .= -1
    model.params.fᵢ .= 1
    s = SimulationSettings(0.01, 0.04, 0.0, 1.0, SerialMode; saveat=[0.01, 0.03, 0.04])
    expected = x .* sum((1 .- 0.98 .^ [1, 3, 4]) ./ 2) ./ 3
    @test s.M == 3
    @test predict(model, s, x; seed=7) ≈ expected
    @test predict(model, s, [0.5]; seed=7) ≈ expected[:, 2:2]
    @test length(ReplayPlan(s; seed=1).seeds) == 1
    @test length(ReplayPlan(SimulationSettings(0.01, 0.04, 0.1, 1.0, 3, ResetMode); seed=1).seeds) == 3

    reset = SimulationSettings(0.01, 0.04, 0.1, 1.0, 1, ResetMode)
    endpoint = SimulationSettings(0.01, 0.04, 0.1, 1.0, SerialMode; saveat=[0.04])
    @test predict(model, endpoint, x; seed=7) ≈ predict(model, reset, x; seed=7)

    initial = fill(0.3, 1, 3)
    start = SimulationSettings(0.01, 0.04, 0.1, 1.0, SerialMode; saveat=[0.0])
    @test ATC._predict!(model, start, initial, x; seed=7) == initial
    @test predict(model, start, x; seed=7) == zero(x)
    r = replay_mse_gradient(model, start, x, zero(x), ReplayPlan(start; seed=1); initial_state=initial)
    @test r.predictions == initial
    @test all(iszero, r.gradient.params.Wᵢⱼ)
    @test all(iszero, r.gradient.params.bᵢ)
    @test only(r.gradient.params.fᵢ) ≈ 2 * 0.3^2

    for invalid in (Float64[], [0.02, 0.01], [0.01, 0.01], [-0.01], [0.05], [NaN])
        @test_throws ArgumentError SimulationSettings(0.01, 0.04, 0.1, 1.0, SerialMode; saveat=invalid)
    end
    @test_throws ArgumentError SimulationSettings(0.01, 0.04, 0.1, 1.0, 3, SerialMode)
    @test_throws ArgumentError SimulationSettings(0.01, 0.04, 0.1, 1.0, 3, SerialMode; saveat=[0.01])
    @test_throws ArgumentError SimulationSettings(0.01, 0.04, 0.1, 1.0, 3, ResetMode; saveat=[0.01])
    times = [0.01, 0.04]
    owned = SimulationSettings(0.01, 0.04, 0.1, 1.0, SerialMode; saveat=times)
    times[1] = 0.02
    @test owned.saveat == [0.01, 0.04]
end

function check_loss_gradient(model, sim, x, y, initial, loss, cache)
    plan = ReplayPlan(sim; seed=739)
    reference(m) = loss(ATC._predict!(m, sim, initial, x; seed=0, trajectory_seeds=plan.seeds), y)
    result = replay_loss_gradient(model, sim, x, y, plan; loss, cache, initial_state=initial)
    @test result.loss ≈ reference(model) rtol=1e-12 atol=1e-14
    for name in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
        values = getproperty(model.params, name)
        for i in eachindex(values)
            original = values[i]
            fd = try
                values[i] = original + 1e-5
                plus = reference(model)
                values[i] = original - 1e-5
                minus = reference(model)
                (plus-minus)/2e-5
            finally
                values[i] = original
            end
            @test getproperty(result.gradient.params, name)[i] ≈ fd rtol=1e-5 atol=1e-8
        end
    end
    repeated = replay_loss_gradient(model, sim, x, y, plan; loss, cache, initial_state=initial)
    @test result == repeated
    return result
end

@testset "Classification and serial gradients against the SDE" begin
    Random.seed!(17)
    x = [0.2 0.8; -0.1 0.3]
    model = TNN(2, 2, 2, 3, x)
    y = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    initial = 0.1 .* randn(4, 2)
    for temp in (0.0, 0.1), mode in (ResetMode, SerialMode)
        sim = mode === ResetMode ? SimulationSettings(0.01, 0.035, temp, 0.7, 3, ResetMode) :
            SimulationSettings(0.01, 0.035, temp, 0.7, SerialMode; saveat=[0.0, 0.013, 0.026])
        cache = ReplayGradientCache(model, sim, x)
        ce = check_loss_gradient(model, sim, x, y, initial, categorical_crossentropy, cache)
        # The same tape must work with a different loss and changing data/parameters.
        check_loss_gradient(model, sim, x, y, initial, mse_loss, cache)
        changed = deepcopy(model)
        changed.params.bᵢ .+= 0.01
        changed.params.fᵢ .*= 1.1
        check_loss_gradient(changed, sim, x .* 0.9, y[[3, 2, 1], :], initial .+ 0.02, categorical_crossentropy, cache)
        @test replay_loss_gradient(model, sim, x, y, ReplayPlan(sim; seed=739);
            loss=categorical_crossentropy, cache, initial_state=initial) == ce
        if mode === SerialMode
            bad = SimulationSettings(0.01, 0.035, temp, 0.7, SerialMode; saveat=[0.0, 0.013, 0.03])
            @test_throws ArgumentError replay_loss_gradient(model, bad, x, y, ReplayPlan(bad; seed=1); cache)
            @test_throws DimensionMismatch replay_mse_gradient(model, sim, x, y, ReplayPlan(sim.M; seed=1); cache)
        end
    end
end

@testset "Classification training and serial iteration snapshots" begin
    x = reshape([-0.5, 0.5], 1, :)
    y = [1.0 0.0; 0.0 1.0]
    for mode in (ResetMode, SerialMode)
        sim = mode === ResetMode ? SimulationSettings(0.01, 0.03, 0.0, 1.0, 2, ResetMode) :
            SimulationSettings(0.01, 0.03, 0.0, 1.0, SerialMode; saveat=[0.01, 0.03])
        Random.seed!(84)
        model, losses, saved = train(1, 1, 2, 2, sim, x, y;
            loss=categorical_crossentropy, maxiters=2, saveat=[0, 1, 2])
        @test losses ≈ [categorical_crossentropy(p, y) for p in saved.predictions[1:2]]
        @test saved.predictions[end] ≈ predict(model, sim, x)
        @test all(isfinite, class_probabilities(saved.predictions[end]))
        for (k, prediction) in zip(saved.iterations, saved.predictions)
            Random.seed!(84)
            reference, _ = train(1, 1, 2, 2, sim, x, y; loss=categorical_crossentropy, maxiters=k)
            @test prediction ≈ predict(reference, sim, x)
        end
        @test_throws ArgumentError train(1, 1, 2, 2, sim, x, y; loss=sin, maxiters=0)
        @test_throws ArgumentError train(1, 1, 2, 2, sim, x, zero(y); loss=categorical_crossentropy, maxiters=0)
    end
    # Float32 conversion must preserve serial observation times and readout shape.
    xf = Float32.(x)
    sf = SimulationSettings(0.01, 0.03, 0.1, 1.0, SerialMode; saveat=[0.01, 0.03])
    mf = TNN(1, 1, 2, 2, xf)
    rf = replay_loss_gradient(mf, sf, xf, Float32.(y), ReplayPlan(sf; seed=1); loss=categorical_crossentropy)
    @test eltype(rf.predictions) == Float32
    @test isfinite(rf.loss)
end
