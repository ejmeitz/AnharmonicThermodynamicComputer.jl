using AnharmonicThermodynamicComputer
using Test, Random

const ATC = AnharmonicThermodynamicComputer

@testset "CPU AD versus finite differences" begin
    Random.seed!(123)
    inputs = randn(2, 3)
    targets = randn(2, 3)
    initial = 0.1 .* randn(4, 3)
    model = TNN(2, 2, 2, 2, inputs)

    for kBT in (0.0, 0.1), M in (1, 3), dt in (0.01, 0.005)
        @testset "kBT=$kBT M=$M dt=$dt" begin
            sim = SimulationSettings(dt, 0.1, kBT, 0.7, M, ResetMode)
            # Reuse exactly the same noise in AD and both FD evaluations.
            plan = ReplayPlan(M; seed=739)
            loss(m) = ATC.mse_loss(m, sim, initial, inputs, targets;
                seed=0, trajectory_seeds=plan.seeds)
            println("Checking AD against FD: kBT=$kBT M=$M dt=$dt")
            flush(stdout)
            result = replay_mse_gradient(model, sim, inputs, targets, plan; initial_state=initial)
            @test result.loss ≈ loss(model)
            max_error = 0.0

            for field in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
                values = getproperty(model.params, field)
                ad = getproperty(result.gradient.params, field)
                # Two perturbation sizes help distinguish FD rounding/truncation
                # error from an incorrect AD gradient. Check every scalar entry.
                for h in (1e-4, 1e-5), i in eachindex(values)
                    original = values[i]
                    fd = try
                        values[i] = original + h
                        plus = loss(model)
                        values[i] = original - h
                        minus = loss(model)
                        (plus - minus) / (2h)
                    finally
                        values[i] = original
                    end
                    max_error = max(max_error, abs(ad[i] - fd))
                    @test isapprox(ad[i], fd; rtol=1e-5, atol=1e-8)
                end
            end
            println("Maximum absolute AD–FD error: $max_error")
        end
    end
end
