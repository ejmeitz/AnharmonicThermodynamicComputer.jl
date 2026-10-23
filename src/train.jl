# https://docs.sciml.ai/SciMLSensitivity/stable/examples/sde/optimization_sde/#Example-3:-Controlling-SDEs-to-an-objective

#! FIGURE THSI FUNC OUT BETTER:
#! NEED TO PASS  SENALG: sensealg = SMS.ForwardDiffSensitivity()
function mse_loss(params, y_target)
    y_pred = _predict!(params)

    yp_sz = size(y_pred) 
    yt_sz = size(y_target)
    # Both should be (n_outputs, n_points)
    @assert yp_sz == yt_sz "Prediction $(yp_sz) and y_target $(yt_sz) shapes must match for MSE loss"

    return sum(abs2, y_pred .- y_target) / yp_sz[2]
end


#! right now hard coded for MSE, need to abstract to other losses 
function train(
    tnn::AllToAllTNN{FT},
    sim_settings::SimulationSettings{FT, SM},
    z::AbstractVecOrMat{FT}, # external forcing
    targets::AbstractVecOrMat{FT}; 
    optimizer=Adam(0.01),
    maxiters::Int = 1000,

) where {FT, SM <: SamplingMode}

    adtype = OPT.AutoZygote() #! which AD backend to use? definitely want reverse mode
    loss_func = (θ, p) -> mse_loss(θ, targets, tnn, sim_settings, z) #! FIGURE OUT HOW TO DEFINE THIS
    optf = OPT.OptimizationFunction(mse_loss, adtype)
    optprob = OPT.OptimizationProblem(optf, params(tnn)) #! ARE PARAMS UPDATED IN PLACE?

    result_sde = OPT.solve(optprob, optimizer; maxiters = maxiters)

end

