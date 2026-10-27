# Keep the model's separate arrays, but expose a flat parameter vector to SciML.
# Packing copies once per prediction, not once per trajectory or force evaluation.
function _pack_parameters(tnn::TNN)
    p = params(tnn)
    return vcat(p.bᵢ, vec(p.Jᵢⱼ), vec(p.Wᵢⱼ), vec(p.fᵢ))
end

# Views also accept AD-tracked numbers without changing the model's storage types.
function _parameter_views(tnn::TNN, values::AbstractVector)
    p = params(tnn)
    nb, nJ, nW, nf = length(p.bᵢ), length(p.Jᵢⱼ), length(p.Wᵢⱼ), length(p.fᵢ)
    length(values) == nb + nJ + nW + nf || throw(DimensionMismatch("incorrect solver parameter count"))

    b = view(values, 1:nb)
    J = reshape(view(values, nb+1:nb+nJ), size(p.Jᵢⱼ))
    W = reshape(view(values, nb+nJ+1:nb+nJ+nW), size(p.Wᵢⱼ))
    f = reshape(view(values, nb+nJ+nW+1:nb+nJ+nW+nf), size(p.fᵢ))

    return (; bᵢ=b, Jᵢⱼ=J, Wᵢⱼ=W, fᵢ=f)
end
