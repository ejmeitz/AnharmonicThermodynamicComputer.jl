
abstract type ThermodynamicNeuron end


# J are couplings, I is input signal
struct QuarticNeuron{P} <: ThermodynamicNeuron
    J₂::Float64
    J₃::Float64
    J₄::Float64
    I::Float64
    coupling::P
end

function QuarticNeuron(J₂, J₃, J₄, I)
    coupling = ImmutablePolynomial([0.0, 0.0, J₂, J₃, J₄])
    return QuarticNeuron(J₂, J₃, J₄, I, coupling)
end

HarmonicNeuron(J₂, I) = QuarticNeuron(J₂, 0.0, 0.0, I)

# Eqn 1
potential_energy(qn::QuarticNeuron, x, b = qn.I) = qn.coupling(x) - (b * x)


#! TODO: add function to compute neuron in thermal equilirbium using Integrals.jl
#! can also get variance (thermal fluctuations) from this