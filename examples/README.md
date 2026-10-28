# Examples

## Learning a cosine

`cosine.jl` implements the nonlinear cosine-learning task in Fig. 2 and
Eqs. (8)–(9) of [Whitelam and Casert's paper](https://doi.org/10.1038/s41467-025-67958-0).
It learns `cos(2πz)` on `[0, 1]` using four layers of eight quadratic-quartic
neurons, zero initial states, reset sampling, `kBT = 0.1`, `μ = 1`, and a final
time of `1`. The full settings use 250 evenly spaced inputs, 1,000 trajectories
per input, and an Euler–Maruyama timestep of `0.001`.

The examples have their own environment with just the local
`AnharmonicThermodynamicComputer` package, `Plots`, and the `Random` standard library
as direct dependencies.
Optimizer and differentiation dependencies are managed by the package itself.
Using Julia 1.11 or later, run from the repository root:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples examples/cosine.jl --smoke
```

This checks one optimizer update and saves the initial and final curves using eight inputs, one
trajectory, and timestep `0.01`. It is not a convergence test.

For the full sampling settings:

```sh
julia --project=examples examples/cosine.jl
```

The current solver runs trajectories serially, so the full run can be expensive
in both time and memory. The example uses the package's Adam/automatic
differentiation trainer rather than the paper's genetic algorithm. Its default
1,000 updates and learning rate `0.01` are example choices; reproducing the
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

## Two-pass gradients

```sh
julia --project=examples examples/replay_gradient.jl
```

This small demonstration computes the gradient of MSE of the mean prediction
using two passes. The first averages trajectories without AD. The second
replays and differentiates one trajectory at a time, accumulating gradients
with the shared residual from the first pass. One ReverseDiff tape is reused
across trajectories. The default `train` and cosine example use this path,
retaining the same tape across all training iterations as well.

The public gradient API is:

```julia
plan = ReplayPlan(sim.M; seed=1234)
result = replay_mse_gradient(tnn, sim, inputs, targets, plan; progress=true)
# result.loss, result.predictions, result.gradient
```

`result.gradient` can be passed to `Optimisers.update!` by callers that use
Optimisers. Apply `enforce_connectivity!` after updating, as the trainer does.
No parameter update occurs inside `replay_mse_gradient`.

Each trajectory has a saved seed and a private RNG, recreated for the second
pass. Reusing a plan reproduces the same noise regardless of unrelated random
draws. Use a new plan seed for fresh noise at the next training iteration.
By default, the prototype checks fingerprints of all saved Brownian values
across passes and throws if they differ. Tests also compare complete histories
exactly and check gradients against finite differences of the forward SDE
ensemble. Replay assumes the same Julia/package versions and simulation
settings; it is not a cross-version randomness guarantee.

Training supports CPU matrices (or vectors for one point), reset sampling,
quartic/harmonic neurons, and fixed-step EM. It uses ReverseDiff directly;
the old `sensealg` training keyword is no longer supported. Each training
iteration draws a fresh trajectory plan from the caller's RNG, so use
`Random.seed!` before training for reproducibility.

### Reusing the AD tape and gradient buffers

```sh
julia --project=examples examples/cached_replay_gradient.jl
```

This example uses the smoke test's eight inputs, four layers of width eight,
`dt=0.01`, and final time `1`, with three trajectories. Construct a cache once,
then reuse it across trajectories and training iterations when computing
gradients yourself (`train` manages this cache automatically):

```julia
cache = ReplayGradientCache(tnn, sim, inputs)
plan = ReplayPlan(sim.M; seed=1234)
result = replay_mse_gradient(tnn, sim, inputs, targets, plan; cache)
# Update tnn with result.gradient, then call again with this cache and a new plan.
```

The cache records an array-based, fixed-step EM calculation and calls
`ReverseDiff.gradient!` with preallocated derivative buffers. Noise increments,
parameters, inputs, initial states, and the output cotangent are explicit tape
inputs; none of their original values are frozen into the recorded calculation.
The SDE solver generates the Brownian histories in both passes. Runtime checks
compare those histories' fingerprints and the tape's predictions to the solver.
Numerical operation order can differ at floating-point roundoff level.

The cache supports changes to parameter values, input/target values, initial
states, seeds, and trajectory count. Rebuild it if array shapes/types, neuron
coefficients, timestep, final time, mobility, or temperature change. Do not share
one cache between concurrent calls. Results own their arrays and remain valid
after subsequent calls. Forward SDE solves and some tape operations still
allocate; this is not a zero-allocation API. Omitting `cache` builds one tape
per gradient call, rather than one per trajectory.

ReverseDiff is a direct dependency of the package. SciMLSensitivity and Zygote
are no longer required; the examples still need only the package, Random, and Plots.
For an existing examples environment, run `using Pkg; Pkg.resolve()` after
updating the package so its manifest reflects the dependency changes.
