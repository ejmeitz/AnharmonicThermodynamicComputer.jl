# Use like OrdinaryDiffEq.jl or something to model this, could also just code it the ODE is not that hard to time-step
abstract type SamplingMode end

struct ResetMode <: SamplingMode end
struct SerialMode <: SamplingMode end


function train!(
    tnn::TNN{FT},
    Δt::FT,
    t_final::FT,
    I::AbstractVector{FT},
    target_function::Function,
    M::Int, # number of samples
    ::Type{SM}
) where {SM <: SamplingMode, FT}

    x₀ = similar(tnn.biases)
    x₀ .= 0.0

    #TODO genetic algo

    samples = _sample!(
        tnn,
        x₀,
        Δt,
        t_final,
        I,
        M,
        SM
    )

    #TODO eval loss
end

function _sample!(
    tnn::TNN{FT, _, V},
    x::FT, 
    Δt::FT,
    t_final::FT,
    I::AbstractVector{FT},
    M::Int, # number of samples
    ::Type{ResetMode}
) where {FT, _, V}

    N_steps = length(0:Δt:t_final)
    N_last = neurons_in_last_layer(tnn)
    res = similar(V, N_last)

    for m in 1:M #! can parallelize over this and only need new x and output for each process
        # Evolve system according to dynamics
        simulate!(tnn, x, Δt, I, N_steps)
        # Sample final layer neurons and reset
        res .+= @view x[end - N_last + 1 : end]
        x .= zero(FT)
    end

end

function _sample!(
    tnn::TNN{FT, _, V},
    x::FT, 
    Δt::FT,
    t_final::FT,
    I::AbstractVector{FT},
    M::Int, # number of samples
    ::Type{SerialMode}
) where {FT, _, V}
    #TODO
end

#TODO Add specialization of this using DifferentialEquations.jl
function simulate!(tnn, x, Δt, I, N_steps)
    for _ in 1:N_steps
        step!(tnn, x, Δt, I)
    end
    return tnn
end

function step!(
    tnn::TNN{FT},
    x::AbstractVector{FT},
    Δt::FT,
    I::AbstractVector{FT}
)

end


function loss()  

end