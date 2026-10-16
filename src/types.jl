

abstract type Oscillator end

# probably find some polynomial library for this...
# will make doing math easier
struct QuarticOscillator # ax^4 + b x^3 + c x^2 + d x + e
    a::T
    b::T
    c::T
    d::T
    e::T
end

# e can be 0 cause it will not effect dynamics nor the
# equilibrium distribution so specifying it doesn't matter
function QuarticOscillator(a, b, c, d, e = 0) where T
    return QuarticOscillator{T}(a, b, c, d, e)
end

struct Neuron
    osc::Oscillator
end


# J_ij*x_i*x_j
# x1 is state of neuron 1
# x2 is state of neuron 2
interact(x1, x2, n1::Neuron, n2::Neuron) = 

# eh probably some way to do this as a matmul instead of node by node
# will write out the for loop and see what operation it is