NORMAL_CONSTANT = sqrt(2*pi)
struct GaussianEstimator
    weight_sum::Float64
    mean::Float64
    variance_sum::Float64
end

function GaussianEstimator()
    return GaussianEstimator(0.0, 0.0, 0.0)
end

function add_weight(g::GaussianEstimator, value::Float64, weight::Float64)
    if g.weight_sum == 0
        return GaussianEstimator(weight, value, 0.0)
    else
        new_weight = g.weight_sum + weight
        new_mean = g.mean + weight * (value - g.mean) / new_weight
        new_variance = g.variance_sum + weight * (value - g.mean) * (value - new_mean)
        return GaussianEstimator(new_weight, new_mean, new_variance)
    end
end

function variance(g::GaussianEstimator)
    return g.weight_sum > 1.0 ? g.variance_sum / (g.weight_sum - 1.0) : 0.0
end
function stdev(g::GaussianEstimator)
    return sqrt(variance(g))
end