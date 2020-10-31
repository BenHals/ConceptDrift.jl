abstract type AbstractSplitCritereon end

struct GiniSplitCritereon <: AbstractSplitCritereon end
struct NumericAttributeObserver{T<:AbstractSplitCritereon}
    distribution_per_class::Dict{Int, GaussianEstimator}
    num_split_suggestions::Int64
    split_critereon::T
end

function NumericAttributeObserver()
    return NumericAttributeObserver(Dict{Int, GaussianEstimator}(), 10, GiniSplitCritereon())
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



"""
Calculate the gini impurity of a split as the weighted sum
of the gini impurity values of each child node.
"""
function get_split_merit(critereon::GiniSplitCritereon, pre_split_distribution, post_split_distribution)
    child_weights = [sum(collect(values(child_dist))) for (child_idx, child_dist) in enumerate(post_split_distribution)]
    total_weight = sum(child_weights)

    gini = 0.0
    for (child_idx, child_dist) in enumerate(post_split_distribution)
        child_relative_weight = child_weights[child_idx] / total_weight
        child_gini = compute_gini(child_dist, child_weights[child_idx])
        gini += child_relative_weight * child_gini
    end

    # Return Inverse to a low gini gets a high merit.
    return 1.0 - gini
end

"""
Compute the gini impurity of a given distribution of class weights.
The squared proportion of each class is subtracted from 1,
so a gini impurity of 0.0 is acheived when only one class is present.
"""
function compute_gini(class_weight_distribution, total_weight)
    gini = 1.0
    if total_weight > 0.0
        for class_weight in values(class_weight_distribution)
            class_proportion = class_weight / total_weight
            gini -= class_proportion^2
        end
    end
    return gini
end

function get_range_of_merit(critereon::GiniSplitCritereon)
    return 1.0
end




