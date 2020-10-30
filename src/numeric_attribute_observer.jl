struct NumericAttributeObserver{F<:Function}
    distribution_per_class::Dict{Int, GaussianEstimator}
    num_split_suggestions::Int64
    get_split_merit::F
end

function NumericAttributeObserver()
    return NumericAttributeObserver(Dict{Int, GaussianEstimator}(), 10, get_gini_split_merit)
end

function update(o::NumericAttributeObserver, x::Float64, y::Int64)
    if !haskey(o.distribution_per_class, y)
        o.distribution_per_class[y] = GaussianEstimator()
    end
    o.distribution_per_class[y] = add_weight(o.distribution_per_class[y], x, 1.0)
    return
end

function range(o::NumericAttributeObserver)
    min_seen = typemax(Float64)
    max_seen = typemin(Float64)
    for estimator in values(o.distribution_per_class)
        min_seen = min(min_seen, estimator.min_seen)
        max_seen = max(max_seen, estimator.max_seen)
    end
    return min_seen, max_seen
end





