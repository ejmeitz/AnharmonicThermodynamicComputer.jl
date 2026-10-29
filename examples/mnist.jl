using AnharmonicThermodynamicComputer
using MLDatasets: MNIST
using Plots
using Random

const MNIST_IMAGE_SIZE = (28, 28)
const MNIST_INPUT_DIM = prod(MNIST_IMAGE_SIZE)
const MNIST_DIGITS = 0:9
const MNIST_N_CLASSES = length(MNIST_DIGITS)
const MNIST_PIXEL_MAX = Float32(typemax(UInt8))
const MNIST_SPLIT_SIZES = (train=60_000, test=10_000)
const MNIST_MOBILITY = 1f0

const MNIST_DEFAULTS = (
    n_train=10000,
    n_test=1000,
    trajectories=8,
    M_eval=100,
    maxiters=100,
    n_layers=2,
    width=16,
    dt=0.005,
    t_final=1.0,
    kBT=0.1,
    seed=1234,
    eval_batch_size=32,
    eval_trajectory_batch_size=10,
)
const MNIST_FIGURE_SETTINGS = (layout=(1, 2), size=(1000, 450), margin=6 * Plots.mm)

function mnist_subset(
    split,
    count,
    rng,
)
    # Keep the full dataset as bytes; convert only the selected images.
    dataset = MNIST(UInt8, split)
    indices = randperm(rng, length(dataset))[1:count]
    inputs = reshape(Float32.(dataset.features[:, :, indices]), MNIST_INPUT_DIM, count) ./ MNIST_PIXEL_MAX
    labels = dataset.targets[indices]
    targets = zeros(Float32, MNIST_N_CLASSES, count)
    for (j, digit) in enumerate(labels)
        targets[digit + 1, j] = 1f0
    end
    return (; inputs, targets, labels, indices)
end

function mnist_evaluate(
    model,
    data;
    dt,
    t_final,
    kBT,
    M_eval,
    eval_batch_size,
    eval_trajectory_batch_size,
    rng,
)
    scores = zeros(Float32, MNIST_N_CLASSES, size(data.inputs, 2))
    for first_image in 1:eval_batch_size:size(data.inputs, 2)
        columns = first_image:min(first_image + eval_batch_size - 1, size(data.inputs, 2))
        inputs = data.inputs[:, columns]
        for first_trajectory in 1:eval_trajectory_batch_size:M_eval
            count = min(eval_trajectory_batch_size, M_eval - first_trajectory + 1)
            sim = SimulationSettings(dt, t_final, kBT, MNIST_MOBILITY, count, ResetMode)
            batch_mean = predict(model, sim, inputs; seed=rand(rng, UInt64))
            @views scores[:, columns] .+= count .* batch_mean
        end
    end
    scores ./= M_eval
    all(isfinite, scores) || error("MNIST evaluation produced non-finite scores")
    predicted_digits = [argmax(column) - 1 for column in eachcol(scores)]
    accuracy = sum(predicted_digits .== data.labels) / length(data.labels)
    loss = categorical_crossentropy(scores, data.targets)
    return (; scores, probabilities=class_probabilities(scores), predicted_digits, accuracy, loss)
end

"""
    learn_mnist(;
        n_train=100,
        n_test=100,
        trajectories=8,
        M_eval=100,
        maxiters=100,
        n_layers=2,
        width=16,
        dt=0.005,
        t_final=1.0,
        kBT=0.1,
        seed=1234,
        eval_batch_size=32,
        eval_trajectory_batch_size=10,
        nthreads=Threads.nthreads(),
    )

Load MNIST with MLDatasets and train on a reproducible random subset of the
official training split. Columns are flattened 28×28 images in [0, 1]; target
row `digit + 1` is one-hot. Every update uses this same supplied input matrix.
The test subset comes from the separate official test split and is only evaluated.

Final train/test estimates average `M_eval` fresh trajectories without AD.
Evaluation is serial and bounded by both evaluation batch sizes. The default
architecture and physical parameters are a starting experiment, not a tuned
MNIST benchmark. Larger training matrices and more workers need larger AD tapes.
"""
function learn_mnist(;
    n_train::Int=MNIST_DEFAULTS.n_train,
    n_test::Int=MNIST_DEFAULTS.n_test,
    trajectories::Int=MNIST_DEFAULTS.trajectories,
    M_eval::Int=MNIST_DEFAULTS.M_eval,
    maxiters::Int=MNIST_DEFAULTS.maxiters,
    n_layers::Int=MNIST_DEFAULTS.n_layers,
    width::Int=MNIST_DEFAULTS.width,
    dt::Real=MNIST_DEFAULTS.dt,
    t_final::Real=MNIST_DEFAULTS.t_final,
    kBT::Real=MNIST_DEFAULTS.kBT,
    seed::Int=MNIST_DEFAULTS.seed,
    eval_batch_size::Int=MNIST_DEFAULTS.eval_batch_size,
    eval_trajectory_batch_size::Int=MNIST_DEFAULTS.eval_trajectory_batch_size,
    nthreads::Int=Threads.nthreads(),
)
    1 <= n_train <= MNIST_SPLIT_SIZES.train || throw(ArgumentError("n_train must be in 1:$(MNIST_SPLIT_SIZES.train)"))
    1 <= n_test <= MNIST_SPLIT_SIZES.test || throw(ArgumentError("n_test must be in 1:$(MNIST_SPLIT_SIZES.test)"))
    all(>(0), (trajectories, M_eval, maxiters, n_layers, width,
        eval_batch_size, eval_trajectory_batch_size)) || throw(ArgumentError("counts must be positive"))
    1 <= nthreads <= Threads.nthreads() || throw(ArgumentError("nthreads exceeds available Julia threads or is nonpositive"))
    sim = SimulationSettings(Float32(dt), Float32(t_final), Float32(kBT), MNIST_MOBILITY, trajectories, ResetMode)
    println("Loading MNIST: $n_train training images, $n_test test images")
    flush(stdout)
    data_rng = Xoshiro(seed)
    train_data = mnist_subset(:train, n_train, data_rng)
    test_data = mnist_subset(:test, n_test, data_rng)

    Random.seed!(seed)
    println("Training MNIST: $trajectories trajectories per update, $maxiters updates")
    model, losses = train(MNIST_INPUT_DIM, n_layers, width, MNIST_N_CLASSES, sim, train_data.inputs, train_data.targets;
        loss=categorical_crossentropy, maxiters, nthreads)
    all(isfinite, losses) || error("MNIST training produced a non-finite loss")

    println("Evaluating train and test subsets: $M_eval trajectories per image (forward only)")
    flush(stdout)
    eval_rng = Xoshiro(rand(UInt64))
    evaluation = (; dt=sim.Δt, t_final=sim.t_final, kBT=Float32(kBT),
        M_eval, eval_batch_size, eval_trajectory_batch_size, rng=eval_rng)
    train_metrics = mnist_evaluate(model, train_data; evaluation...)
    test_metrics = mnist_evaluate(model, test_data; evaluation...)
    println("Train subset: accuracy=$(round(100train_metrics.accuracy; digits=1))%, cross-entropy=$(train_metrics.loss)")
    println("Test subset: accuracy=$(round(100test_metrics.accuracy; digits=1))%, cross-entropy=$(test_metrics.loss)")

    confusion = zeros(Int, MNIST_N_CLASSES, MNIST_N_CLASSES)
    for (actual, predicted) in zip(test_data.labels, test_metrics.predicted_digits)
        confusion[actual + 1, predicted + 1] += 1
    end
    history = plot(eachindex(losses), losses; xlabel="Optimizer update",
        ylabel="Cross-entropy", label="Training loss", title="MNIST learning", marker=:circle)
    matrix_plot = heatmap(MNIST_DIGITS, MNIST_DIGITS, confusion; xlabel="Predicted digit", ylabel="True digit",
        xticks=MNIST_DIGITS, yticks=MNIST_DIGITS, title="Test subset confusion", aspect_ratio=1)
    figure = plot(history, matrix_plot; MNIST_FIGURE_SETTINGS...)
    return (; model, losses, train_data, test_data, train_metrics, test_metrics, confusion, figure)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = if isempty(ARGS)
        learn_mnist()
    else
        error("Usage: julia --project=examples examples/mnist.jl")
    end
    output = joinpath(@__DIR__, "mnist.png")
    savefig(result.figure, output)
    println("Saved plot to ", output)
end
