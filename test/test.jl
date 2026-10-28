using AnharmonicThermodynamicComputer
using Optimisers
using LinearAlgebra

const ATC = AnharmonicThermodynamicComputer

# Network architecture
FT = Float64

n_inputs = 5
n_layers = 2
neurons_per_layer = 2
n_outputs = 2

n_neurons = n_layers * neurons_per_layer
n_points = 3  # Number of simultaneous input samples

# Simulation settings
Δt = 0.01
t_final = 1.0
kBT = 1.0
μ = 1.0
M = 1  # Start with one trajectory for debugging AD

sim = ATC.SimulationSettings(Δt, t_final, kBT, μ, M, ResetMode)

# External forcing: (n_inputs, n_points)
I = randn(FT, n_inputs, n_points)

# Initial state: (n_neurons, n_points)
X₀ = zeros(FT, n_neurons, n_points)

# Targets: (n_outputs, n_points)
targets = randn(FT, n_outputs, n_points)

# Initialize network
params = TNNParameters(
    n_inputs,
    n_layers,
    neurons_per_layer,
    n_outputs,
    I,
)

tnn = TNN(params, n_neurons, n_layers)

# Diagnostics: verify trainable fields
println("Trainable parameters:")
total = sum(length, Optimisers.trainables(tnn))

for (path, p) in Optimisers.trainables(tnn; path=true)
    println("  $path: $(size(p)) ($(length(p)) elements)")
end

println("Total parameters: $total")

# Reusable gradient tape and fixed trajectory noise
cache = ReplayGradientCache(tnn, sim, I)
plan = ReplayPlan(M; seed=1234)

# Loss function
loss(tnn) = ATC.mse_loss(
    tnn, sim, X₀, I, targets; seed=0, trajectory_seeds=plan.seeds
)

# Test forward evaluation
println("\nTesting forward pass...")
value = loss(tnn)
@show value

# Test gradients
println("\nTesting AD...")
result = replay_mse_gradient(tnn, sim, I, targets, plan; cache, initial_state=X₀)
value, grad = result.loss, result.gradient

@show value

# Check gradients of each trainable array
println("\nGradient diagnostics:")

for name in (:bᵢ, :Jᵢⱼ, :Wᵢⱼ, :fᵢ)
    g = getproperty(grad.params, name)

    if g === nothing
        println("$name: NO GRADIENT")
        continue
    end

    println("$name:")
    @show size(g)
    @show norm(g)
    @show all(isfinite, g)
end

# Test one optimizer update
println("\nTesting optimizer update...")

opt_state = Optimisers.setup(
    Optimisers.Adam(0.01),
    tnn
)

J_before = copy(tnn.params.Jᵢⱼ)
opt_state, tnn_new = Optimisers.update(
    opt_state, tnn, grad
)
enforce_connectivity!(tnn_new)

println("Optimizer update completed.")

@show maximum(abs.(tnn_new.params.Jᵢⱼ .- J_before))
