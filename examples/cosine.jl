using AnharmonicThermodynamicComputer
using Plots
using Random

"""
    learn_cosine(; n_points=250, trajectories=1000, dt=0.001,
                   maxiters=1000, seed=1234,
                   saveat=unique([0, maxiters ÷ 2, maxiters]))

Learn y(z) = cos(2πz), z ∈ [0, 1], using the nonlinear network in Fig. 2 of
Whitelam and Casert, https://doi.org/10.1038/s41467-025-67958-0.
Uses this package's Adam/AD training instead of the paper's genetic algorithm.
Save curves after the specified numbers of training updates. Returns the trained
network, loss history, inputs, targets, saved curves, last saved predictions, and plot.
"""
function learn_cosine(;
    n_points::Int=30,
    trajectories::Int=1000,
    dt::Float64=0.005,
    maxiters::Int=100,
    seed::Int=1234,
    saveat=unique([0, maxiters ÷ 2, maxiters]),
)
    n_points >= 2 || throw(ArgumentError("n_points must be at least 2"))
    trajectories >= 1 || throw(ArgumentError("trajectories must be positive"))
    maxiters >= 1 || throw(ArgumentError("maxiters must be positive"))
    isfinite(dt) && 0 < dt <= 1 || throw(ArgumentError("dt must be in (0, 1]"))
    isempty(saveat) && throw(ArgumentError("saveat must contain at least one iteration to plot"))
    Random.seed!(seed)

    # One scalar input and output per column, including both endpoints.
    inputs = reshape(collect(range(0.0, 1.0; length=n_points)), 1, n_points)
    targets = cospi.(2 .* inputs)

    # Paper: width 8, depth 4, J₂=1, J₃=0, J₄=1 (the TNN defaults),
    # β=10 (kBT=0.1), μ=1, observation time 1, reset from zero each time.
    n_layers, width = 4, 8
    sim = SimulationSettings(dt, 1.0, 0.1, 1.0, trajectories, ResetMode)

    println("Learning cos(2πz): $n_points points, $trajectories trajectories, $maxiters updates")
    flush(stdout)
    tnn, losses, saved = train(
        1, n_layers, width, 1, sim, inputs, targets;
        maxiters, saveat,
    )
    all(isfinite, losses) || error("Training produced a non-finite loss")

    # Training already retained the requested curves; no separate solve is needed here.
    all(p -> all(isfinite, p), saved.predictions) || error("Training saved a non-finite prediction")
    predictions = last(saved.predictions)
    mse = sum(abs2, predictions .- targets) / n_points
    println("Initial training MSE: ", first(losses))
    println("Last training MSE (before final update): ", last(losses))
    println("Saved prediction MSE after $(last(saved.iterations)) updates: ", mse)
    println("A short run checks execution; it does not establish convergence.")

    curve = plot(
        vec(inputs), vec(targets);
        label="Target cos(2πz)", xlabel="z", ylabel="y", linewidth=2,
        title="Cosine learning",
    )
    for (iteration, prediction) in zip(saved.iterations, saved.predictions)
        plot!(curve, vec(inputs), vec(prediction); label="After $iteration updates", linewidth=2)
    end
    history = plot(
        eachindex(losses), losses;
        label="Training MSE", xlabel="Optimizer update", ylabel="MSE",
        title="Training loss", linewidth=2, marker=:circle,
    )
    figure = plot(curve, history; layout=(1, 2), size=(1000, 400))

    return (; tnn, losses, inputs, targets, predictions, saved, mse, figure)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = if isempty(ARGS)
        learn_cosine()
    elseif ARGS == ["--smoke"]
        # Keep the paper's architecture and temperature, but reduce runtime.
        learn_cosine(; n_points=8, trajectories=1, dt=0.01, maxiters=1)
    else
        error("Usage: julia --project=examples examples/cosine.jl [--smoke]")
    end
    output = joinpath(@__DIR__, "cosine.png")
    savefig(result.figure, output)
    println("Saved plot to ", output)
end
