
abstract type ThermodynamicNeuron{FT <: AbstractFloat} end


struct QuarticNeuron{FT} <: ThermodynamicNeuron{FT}
    J₂::FT
    J₃::FT
    J₄::FT
end

HarmonicNeuron(J₂::FT) where {FT <: AbstractFloat} =
    QuarticNeuron(J₂, zero(FT), zero(FT))

# U(x) = J₂x² + J₃x³ + J₄x⁴ - bx
potential_energy(qn::QuarticNeuron, x, b) =
    x^2 * (qn.J₂ + x * (qn.J₃ + x * qn.J₄)) - b * x

# F(x) = -2J₂x - 3J₃x² - 4J₄x³ + b
force(qn::QuarticNeuron, x, b) =
    b - x * (2 * qn.J₂ + x * (3 * qn.J₃ + 4 * qn.J₄ * x))


# Average neuron output in thermal equilibrium at temperature `T`
function average_equilibrium_output(
    tn::ThermodynamicNeuron{FT},
    I::AbstractVector{FT},
    kBT::Real;
    n_pts = 128
) where FT
    
    β = inv(FT(kBT))
    points, w = gausshermite(FT, n_pts)
    logweights = @. log(w) + x^2 - (β * potential_energy(Ref(tn), points, I))
    weights = exp.(logweights .- maximum(logweights)) # try to prevent overflow
    return dot(x, weights) / sum(weights)
end


