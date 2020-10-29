using Distributions

NORMAL_CONSTANT = sqrt(2*pi)

struct GaussianEstimator
    weight_sum::Float64
    mean::Float64
    variance_sum::Float64
    min_seen::Float64
    max_seen::Float64
end

function GaussianEstimator()
    return GaussianEstimator(0.0, 0.0, 0.0, typemax(Float64), typemin(Float64))
end

function add_weight(g::GaussianEstimator, value::Float64, weight::Float64)
    if g.weight_sum == 0
        return GaussianEstimator(weight, value, 0.0, value, value)
    else
        new_weight = g.weight_sum + weight
        new_mean = g.mean + weight * (value - g.mean) / new_weight
        new_variance = g.variance_sum + weight * (value - g.mean) * (value - new_mean)
        new_min = min(g.min_seen, value)
        new_max = max(g.max_seen, value)
        return GaussianEstimator(new_weight, new_mean, new_variance, new_min, new_max)
    end
end

function variance(g::GaussianEstimator)
    return g.weight_sum > 1.0 ? g.variance_sum / (g.weight_sum - 1.0) : 0.0
end
function stdev(g::GaussianEstimator)
    return sqrt(variance(g))
end

function pdf(g::GaussianEstimator, value::Float64)
    if g.weight_sum > 0.0
        std_dev = stdev(g)
        if std_dev > 0.0
            diff = value - g.mean
            return ((1.0 / (NORMAL_CONSTANT * std_dev)) * exp(-(diff^2 / (2.0 * std_dev^2))))
        elseif value == g.mean
            return 1.0
        end
    end
    return 0.0
end
    
"""
Estimates the amount of weight seen for a given class value for attribute 
values less than, equal to and greater than the given value.

This is used to estimate the future distribution if a split point is added
at the value.
"""
function estimated_weight_split(g::GaussianEstimator, value::Float64)
    equalto_weight = pdf(g, value) * g.weight_sum
    std_dev = stdev(g)
    if std_dev > 0.0
        lessthan_weight = (cdf(Normal(g.mean, std_dev), value) * g.weight_sum) - equalto_weight
    else
        if value < g.mean
            lessthan_weight = g.weight_sum - equalto_weight
        else
            lessthan_weight = 0.0
        end
    end
    greaterthan_weight = max(g.weight_sum - equalto_weight - lessthan_weight, 0.0)

    return (lessthan_weight, equalto_weight, greaterthan_weight)
end
