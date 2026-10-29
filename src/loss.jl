export mse_loss, categorical_crossentropy, class_probabilities

function _check_loss_shapes(predictions, targets)
    size(predictions) == size(targets) || throw(DimensionMismatch("predictions and targets must have the same shape"))
    !isempty(predictions) || throw(ArgumentError("loss requires nonempty predictions"))
end

"""Mean squared error, summed over outputs and averaged over input columns."""
function mse_loss(predictions::AbstractMatrix, targets::AbstractMatrix)
    _check_loss_shapes(predictions, targets)
    return sum(abs2, predictions .- targets) / size(predictions, 2)
end

"""
    class_probabilities(logits)

Stable softmax over classes (rows), independently for each input column.
Pass the averaged scores returned by `predict`, not individual trajectory scores.
"""
function class_probabilities(logits::AbstractMatrix)
    !isempty(logits) && all(isfinite, logits) || throw(ArgumentError("logits must be finite and nonempty"))
    shifted = logits .- maximum(logits; dims=1)
    probabilities = exp.(shifted)
    probabilities ./= sum(probabilities; dims=1)
    return probabilities
end

function _check_categorical_targets(logits, targets)
    _check_loss_shapes(logits, targets)
    all(isfinite, logits) || throw(ArgumentError("logits must be finite"))
    all(t -> isfinite(t) && t >= 0, targets) || throw(ArgumentError("class targets must be finite and nonnegative"))
    FT = float(eltype(logits))
    all(s -> isapprox(s, one(FT); rtol=sqrt(eps(FT)), atol=zero(FT)), sum(targets; dims=1)) ||
        throw(ArgumentError("each target column must sum to one; supply one-hot labels or class probabilities"))
end

"""
    categorical_crossentropy(logits, targets)

Cross-entropy from raw class scores and one-hot (or soft probability) target
columns, averaged over inputs. Uses log-softmax for numerical stability; do not
apply softmax yourself. For MNIST, arrays have shape `(10, n_images)` and row
`digit + 1` identifies the class.
"""
function categorical_crossentropy(logits::AbstractMatrix, targets::AbstractMatrix)
    _check_categorical_targets(logits, targets)
    shifted = logits .- maximum(logits; dims=1)
    logprobabilities = shifted .- log.(sum(exp.(shifted); dims=1))
    return -sum(targets .* logprobabilities) / size(logits, 2)
end

function _loss_and_cotangent!(cotangent, ::typeof(mse_loss), predictions, targets)
    value = mse_loss(predictions, targets)
    cotangent .= (2 / size(predictions, 2)) .* (predictions .- targets)
    return value
end

function _loss_and_cotangent!(cotangent, ::typeof(categorical_crossentropy), predictions, targets)
    value = categorical_crossentropy(predictions, targets)
    probabilities = class_probabilities(predictions)
    # Include target column sums so this remains the derivative of the actual
    # loss even when floating-point probability columns sum only approximately to 1.
    cotangent .= (probabilities .* sum(targets; dims=1) .- targets) ./ size(predictions, 2)
    return value
end

function _check_loss(loss)
    loss in (mse_loss, categorical_crossentropy) ||
        throw(ArgumentError("loss must be mse_loss or categorical_crossentropy"))
end
