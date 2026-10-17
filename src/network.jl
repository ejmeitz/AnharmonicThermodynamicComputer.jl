struct ThermodynamicNeuralNetwork{L}
    layers::L
    Wᵢⱼ::AbstractMatrix{Float64} # external coupling
end

function potential_energy(
    n::ThermodynamicNeuralNetwork,
    x::AbstractVector,
    I::AbstractVector
)


end