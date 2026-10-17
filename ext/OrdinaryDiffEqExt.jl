## In future could be cool to define StochasticDiffEq problem
# and use their integrators instead of the forward Euler step

module OrdinaryDiffEqExt

using AnharmonicThermodynamicComputer

# Define specialization for DifferentialEquations.jl types
function AnharmonicThermodynamicComputer.simulate!()

end

end