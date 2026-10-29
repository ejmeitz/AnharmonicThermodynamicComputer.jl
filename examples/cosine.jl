using AnharmonicThermodynamicComputer
using Plots
using Random

const COSINE_DEFAULTS = (
    n_points=30,
    trajectories=250,
    dt=0.005,
    maxiters=250,
    seed=1234,
)
const COSINE_N_INPUTS = 1
const COSINE_N_OUTPUTS = 1
const COSINE_N_LAYERS = 4
const COSINE_WIDTH = 8
const COSINE_FINAL_TIME = 1.0
const COSINE_TEMPERATURE = 0.1
const COSINE_MOBILITY = 1.0
const COSINE_EVAL_TRAJECTORY_MULTIPLIER = 4
const COSINE_EVAL_BATCH_SIZE = 10
const COSINE_FIGURE_SETTINGS = (layout=(1, 2), size=(1000, 400))

"""
    learn_cosine(;
        n_points=30,
        trajectories=250,
        dt=0.005,
        maxiters=250,
        seed=1234,
        eval_trajectories=4*trajectories,
        saveat=unique([0, maxiters ÷ 2, maxiters]),
        train_kwargs...,
    )

Learn y(z) = cos(2πz), z ∈ [0, 1], using the nonlinear network in Fig. 2 of
Whitelam and Casert, https://doi.org/10.1038/s41467-025-67958-0.
Uses this package's Adam/AD training instead of the paper's genetic algorithm.
Additional keywords are forwarded to `train`, including `nthreads`, which
defaults to `Threads.nthreads()`. Each worker has one private AD cache.
Save curves after the specified numbers of training updates. Returns the trained
network, loss history, inputs, targets, saved curves, last saved predictions, and plot.
Set `eval_trajectories` to a positive trajectory count for an additional forward-only
estimate of the final trained model. This adds a curve to the plot and makes
`predictions` and `mse` refer to that estimate; `saved` keeps the training snapshots.
Evaluation uses fresh noise in batches bounded by `COSINE_EVAL_BATCH_SIZE`.
"""
function learn_cosine(;
    n_points::Int=COSINE_DEFAULTS.n_points,
    trajectories::Int=COSINE_DEFAULTS.trajectories,
    dt::Float64=COSINE_DEFAULTS.dt,
    maxiters::Int=COSINE_DEFAULTS.maxiters,
    seed::Int=COSINE_DEFAULTS.seed,
    eval_trajectories::Union{Nothing,Int}=COSINE_EVAL_TRAJECTORY_MULTIPLIER * trajectories,
    saveat=unique([0, maxiters ÷ 2, maxiters]),
    train_kwargs...,
)
    n_points >= 2 || throw(ArgumentError("n_points must be at least 2"))
    trajectories >= 1 || throw(ArgumentError("trajectories must be positive"))
    maxiters >= 1 || throw(ArgumentError("maxiters must be positive"))
    eval_trajectories === nothing || eval_trajectories > 0 || throw(ArgumentError("eval_trajectories must be positive"))
    isfinite(dt) && 0 < dt <= COSINE_FINAL_TIME || throw(ArgumentError("dt must be in (0, $COSINE_FINAL_TIME]"))
    isempty(saveat) && throw(ArgumentError("saveat must contain at least one iteration to plot"))
    Random.seed!(seed)

    # One scalar input and output per column, including both endpoints.
    inputs = reshape(collect(range(0.0, 1.0; length=n_points)), COSINE_N_INPUTS, n_points)
    targets = cospi.(2 .* inputs)

    # Paper: width 8, depth 4, J₂=1, J₃=0, J₄=1 (the TNN defaults),
    # β=10 (kBT=0.1), μ=1, observation time 1, reset from zero each time.
    sim = SimulationSettings(dt, COSINE_FINAL_TIME, COSINE_TEMPERATURE, COSINE_MOBILITY, trajectories, ResetMode)

    println("Learning cos(2πz): $n_points points, $trajectories trajectories, $maxiters updates")
    flush(stdout)
    tnn, losses, saved = train(
        COSINE_N_INPUTS, COSINE_N_LAYERS, COSINE_WIDTH, COSINE_N_OUTPUTS, sim, inputs, targets;
        maxiters, saveat, train_kwargs...,
    )
    all(isfinite, losses) || error("Training produced a non-finite loss")

    # Retain the training snapshots even when requesting a more precise final estimate.
    all(p -> all(isfinite, p), saved.predictions) || error("Training saved a non-finite prediction")
    predictions = last(saved.predictions)
    if eval_trajectories !== nothing
        println("Evaluating trained model with $eval_trajectories fresh trajectories (forward only)")
        flush(stdout)
        eval_rng = Xoshiro(rand(UInt64))
        predictions = zeros(size(targets))
        for first_index in 1:COSINE_EVAL_BATCH_SIZE:eval_trajectories
            count = min(COSINE_EVAL_BATCH_SIZE, eval_trajectories - first_index + 1)
            eval_sim = SimulationSettings(dt, COSINE_FINAL_TIME, COSINE_TEMPERATURE, COSINE_MOBILITY, count, ResetMode)
            batch_mean = predict(tnn, eval_sim, inputs; seed=rand(eval_rng, UInt64))
            predictions .+= count .* batch_mean
        end
        predictions ./= eval_trajectories
        all(isfinite, predictions) || error("Final evaluation produced a non-finite prediction")
    end
    mse = sum(abs2, predictions .- targets) / n_points
    println("Initial training MSE: ", first(losses))
    println("Last training MSE (before final update): ", last(losses))
    if eval_trajectories === nothing
        println("Saved prediction MSE after $(last(saved.iterations)) updates: ", mse)
    else
        println("Final evaluation MSE ($eval_trajectories trajectories, $maxiters updates): ", mse)
    end
    println("A short run checks execution; it does not establish convergence.")

    curve = plot(
        vec(inputs), vec(targets);
        label="Target cos(2πz)", xlabel="z", ylabel="y", linewidth=2,
        title="Cosine learning",
    )
    for (iteration, prediction) in zip(saved.iterations, saved.predictions)
        plot!(curve, vec(inputs), vec(prediction); label="After $iteration updates", linewidth=2)
    end
    if eval_trajectories !== nothing
        plot!(curve, vec(inputs), vec(predictions);
            label="Final estimate ($eval_trajectories trajectories)", linewidth=3, linestyle=:dash)
    end
    history = plot(
        eachindex(losses), losses;
        label="Training MSE", xlabel="Optimizer update", ylabel="MSE",
        title="Training loss", linewidth=2, marker=:circle,
    )
    figure = plot(curve, history; COSINE_FIGURE_SETTINGS...)

    return (; tnn, losses, inputs, targets, predictions, saved, mse, figure, eval_trajectories)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = if isempty(ARGS)
        learn_cosine()
    else
        error("Usage: julia --project=examples examples/cosine.jl")
    end
    output = joinpath(@__DIR__, "cosine.png")
    savefig(result.figure, output)
    println("Saved plot to ", output)
end
