# Examples

## Learning a cosine

`cosine.jl` implements the nonlinear cosine-learning task in Fig. 2 and
Eqs. (8)–(9) of [Whitelam and Casert's paper](https://doi.org/10.1038/s41467-025-67958-0).
It learns `cos(2πz)` on `[0, 1]` using four layers of eight quadratic-quartic
neurons, zero initial states, reset sampling, `kBT = 0.1`, `μ = 1`, and a final
time of `1`. The defaults use 30 evenly spaced inputs, 250 trajectories
per input, and an Euler–Maruyama timestep of `0.005`.

The examples environment includes the local `AnharmonicThermodynamicComputer`
package, `Plots`, `Random`, and `MLDatasets` for the MNIST example.
Optimizer and differentiation dependencies are managed by the package itself.
Using Julia 1.11 or later, run from the repository root:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

For the full sampling settings:

```sh
julia --project=examples examples/cosine.jl
```

Training uses the configured Julia thread count (see below). The example uses
the package's Adam/automatic differentiation trainer rather than the paper's
genetic algorithm. Its default 250 updates and learning rate `0.01` are example choices; reproducing the
paper's fitted curve is not guaranteed.

For a custom run, start the Julia REPL with `julia --project=examples`:

```julia
include("examples/cosine.jl")
result = learn_cosine(; n_points=32, trajectories=4, maxiters=100, saveat=[0, 10, 50, 100])
# result.inputs, result.targets, result.predictions are 1 × n_points matrices.
# result.tnn and result.losses contain the trained model and training history.
# result.saved.iterations and result.saved.predictions contain the saved curves.
display(result.figure)
```

For a more precise final estimate without backpropagation, set `M_eval` on the
cosine example:

```julia
result = learn_cosine(; trajectories=25, maxiters=250, M_eval=1000)
```

Training still uses 25 trajectories per update. After training, the example
averages 1,000 fresh trajectories using public `predict` calls in batches of at
most 10, then plots the final estimate. `result.predictions` and `result.mse`
refer to this estimate; `result.saved` still contains the training snapshots.
No gradients or optimizer updates occur during evaluation. Omitting `M_eval`
keeps the existing behavior. This option belongs only to `learn_cosine`.

The command-line script saves `examples/cosine.png`, showing the target and
learned curves alongside training loss, and reports training and saved-prediction
MSE. `saveat` specifies completed optimizer updates, not simulation times. By
default the example saves iteration 0, the midpoint, and the final iteration.

The public training API returns the requested curves directly:

```julia
tnn, losses, saved = train(1, 4, 8, 1, sim, inputs, targets;
                          maxiters=100, saveat=[0, 10, 50, 100])
```

Each entry in `saved.predictions` is a matrix corresponding to the same entry
in `saved.iterations`. Intermediate snapshots reuse training predictions;
saving the final iteration performs one final forward evaluation with fresh
noise. Snapshots are returned in memory. Omitting `saveat` preserves the
two-value `(tnn, losses)` return. For new inputs after training, use
`predict(tnn, sim, new_inputs; seed=1234)`.

## Replay gradients

`train` manages reusable ReverseDiff tapes internally. Custom loops can use
`ReplayGradientCache`, `ReplayPlan`, and `replay_loss_gradient` directly; the
standalone replay benchmark examples have been removed. The tests under `test/`
cover gradients, cache reuse, and deterministic Brownian replay.

### Threaded training

Start Julia with multiple threads:

```sh
julia --threads=auto --project=examples
```

Training uses Julia's configured thread count by default. Limit it when needed:

```julia
include("examples/cosine.jl")
result = learn_cosine(; nthreads=4, maxiters=500)
```

The `nthreads` keyword belongs to `train`; `learn_cosine` forwards extra keywords
to it. Its default is `Threads.nthreads()`, and `nthreads=1` selects serial
execution. The requested count cannot exceed Julia's available threads.
`trajectory_batch_size` defaults to `nthreads`; setting it lower further limits
concurrency and memory. Both solves and AD respect these limits.
This uses `EnsembleThreads` for bounded
groups of reset trajectories, then parallel backpropagation with a private
reusable tape per worker. The first pass still computes the complete mean
prediction; batching does not change the loss or introduce extra optimizer
updates. Each trajectory keeps its explicit seed, and gradients are summed in
trajectory order. Final training snapshots also use bounded forward ensembles.

`SerialMode` follows one continuous trajectory and automatically uses one worker.
Ordinary `predict` retains
its existing execution path. Worker tapes require more memory: four workers
retain roughly four times the single-tape storage. Saved trajectory histories
are bounded by the smaller of `nthreads` and `trajectory_batch_size`, including
partial final groups. Using all configured threads increases tape memory accordingly.
Benchmarks used four Julia threads with one BLAS thread; concurrent BLAS threads
can compete with trajectory workers for the same CPU cores.

For custom loops, construct and reuse the threaded cache explicitly:

```julia
cache = ThreadedReplayGradientCache(model, sim, inputs; batch_size=10)
result = replay_loss_gradient(model, sim, inputs, targets,
                              ReplayPlan(sim; seed=1234); cache)
```

The cache accepts an optional `workers` count to limit both solver concurrency
and the number of AD tapes; it must not exceed the number of Julia threads.
Its effective batch size cannot exceed `workers`. Neither cache type may be shared
between concurrent training calls. The default serial cache and MSE wrapper
remain supported.

ReverseDiff is a direct dependency of the package. SciMLSensitivity and Zygote
are no longer required; MLDatasets is needed only for the MNIST example.
For an existing examples environment, run `using Pkg; Pkg.resolve()` after
updating the package so its manifest reflects the dependency changes.

## MNIST

[`MLDatasets.MNIST`](https://juliaml.github.io/MLDatasets.jl/stable/datasets/vision/)
loads the official training and test splits. MLDatasets is an examples-only
dependency; the core package does not depend on a dataset loader.

From the repository root:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/mnist.jl
```

If an older examples manifest causes a dependency conflict, run
`using Pkg; Pkg.add("MLDatasets")` in that environment to resolve compatible
versions. On first use, MLDatasets prompts to download MNIST to its dataset cache.
The script uses `MNIST_DEFAULTS` and saves `examples/mnist.png`.

For a configurable run:

```julia
include("examples/mnist.jl")
result = learn_mnist(; n_train=100, n_test=100, trajectories=8,
    M_eval=100, maxiters=100, nthreads=Threads.nthreads())
display(result.figure)
result.test_metrics.accuracy
result.test_metrics.probabilities  # 10 x n_test; row digit + 1
```

Images become Float32 columns of length 784 with values in `[0, 1]`, and targets
are one-hot columns with row `digit + 1`. Training uses
`loss=categorical_crossentropy` on trajectory-averaged scores. Softmax is applied
only afterward to report probabilities.

`n_train` is the size of the fixed, randomly selected input matrix supplied to
`train`. Every optimizer update uses that same matrix; this example does not
cycle through all 60,000 images. The test subset is drawn from the separate test
split and never used for optimization. `seed` controls subset selection and
training initialization. Dataset loading keeps the full images as bytes and
converts only the selected subset to Float32.

After training, both subsets get forward-only estimates using `M_eval`
trajectories per image. `eval_batch_size=32` bounds images per evaluation call,
and `eval_trajectory_batch_size=10` bounds trajectories per call. These calls
are serial and use fresh noise. `train_metrics` and `test_metrics` contain
scores, probabilities, predicted digits, accuracy, and cross-entropy. The figure
shows training loss and the test confusion matrix.

The default two layers of width 16, timestep 0.005, final time 1, and temperature
0.1 are starting settings, not a tuned MNIST benchmark. Increase training subset
size and worker count gradually: both affect AD memory. A tiny subset can be
used to check whether the model can fit its training data before scaling up.

For a custom training loop, use the same replay cache with either loss:

```julia
cache = ReplayGradientCache(model, sim, inputs)
plan = ReplayPlan(sim; seed=1234)
result = replay_loss_gradient(model, sim, inputs, targets, plan;
                              loss=categorical_crossentropy, cache)
```

`replay_mse_gradient` remains available as the MSE convenience wrapper.

## Serial sampling

Serial mode keeps each input fixed and takes all observations along one
continuous trajectory starting at zero. `predict` returns their arithmetic mean,
not just the final state. Specify a vector or range of **simulation times** in
the settings:

```julia
inputs = reshape(collect(range(0.0, 1.0; length=8)), 1, :)
targets = cospi.(2 .* inputs)
serial = SimulationSettings(0.005, 1.0, 0.1, 1.0, SerialMode;
                            saveat=0.2:0.2:1.0)
# serial.M == 5 observations, from one trajectory per input.
model, losses, snapshots = train(1, 4, 8, 1, serial, inputs, targets;
                                 maxiters=100, saveat=[0, 50, 100])
prediction = predict(model, serial, inputs; seed=1234)
```

Here the `SimulationSettings` keyword specifies observation times, while the
`train` keyword still specifies **optimizer iterations** at which to retain mean
predictions. The six-positional-argument settings constructor also accepts
`SerialMode; saveat=times`, provided its `M` equals `length(times)`.
Observation times must be finite, sorted, unique, and within `[0, t_final]`.
They may be irregular or include zero; the integrator steps exactly to each
observation rather than interpolating between steps.

`ReplayPlan(serial; seed=1234)` contains one seed for the entire path. Replay
differentiates through every requested observation and the intervening dynamics;
states are not reset or detached at sample times. MSE and cross-entropy both
support this mode. Serial observations can be correlated, so their count is not
an equivalent number of independent trajectories. Tape and noise-history memory
grow with the length of the continuous trajectory; this implementation does not
yet checkpoint long trajectories. Start with a short horizon.
