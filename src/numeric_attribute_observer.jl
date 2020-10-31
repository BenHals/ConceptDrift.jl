abstract type AbstractSplitCritereon end

struct GiniSplitCritereon <: AbstractSplitCritereon end
struct NumericAttributeObserver{T<:AbstractSplitCritereon}
    distribution_per_class::Dict{Int, GaussianEstimator}
    num_split_suggestions::Int64
    split_critereon::T
end

abstract type AbstractConditionalTest end

struct NumericBinaryTest <: AbstractConditionalTest
    attr_idx::Int64
    split_value::Float64
end


struct AttributeSplitSuggestion
    split_test::AbstractConditionalTest
    resulting_class_distribution::Tuple{Dict{Int64, Float64}, Dict{Int64, Float64}}
    merit::Float64
    num_splits::Int64
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

"""
Propose values an attribute could be split at, between the min and max seen.
"""
function get_split_point_suggestions(o::NumericAttributeObserver)
    min_seen, max_seen = range(o)
    split_difference = (max_seen - min_seen) / (o.num_split_suggestions + 1)
    split_values = Vector{Float64}()
    for i in 1:o.num_split_suggestions
        split_value = min_seen + (split_difference * i)
        if split_value > min_seen
            push!(split_values, split_value)
        end
    end
    return split_values
end


function get_binary_split_class_distribution(o::NumericAttributeObserver, split_value::Float64)
    lhs_dist = Dict{Int64, Float64}()
    rhs_dist = Dict{Int64, Float64}()
    for (class, estimator) in o.distribution_per_class
        if split_value < estimator.min_seen
            rhs_dist[class] = estimator.weight_sum
        elseif split_value > estimator.max_seen
            lhs_dist[class] = estimator.weight_sum
        else
            l_dist, e_dist, r_dist = estimated_weight_split(estimator, split_value)
            lhs_dist[class] = l_dist+e_dist
            rhs_dist[class] = r_dist
        end
    end
    return (lhs_dist, rhs_dist)
end

function get_best_split_suggestion(o::NumericAttributeObserver, pre_split_distribution::Dict{Int64, Float64}, att_idx::Int64)
    best_suggestion = (typemin(Float64), 0.0, (Dict{Int64, Float64}(), Dict{Int64, Float64}()))
    suggested_split_values = get_split_point_suggestions(o)
    for split_value in suggested_split_values
        post_split_dist = get_binary_split_class_distribution(o, split_value)
        merit = get_split_merit(o.split_critereon, pre_split_distribution, post_split_dist)
        if merit > best_suggestion[1]
            best_suggestion = (merit, split_value, post_split_dist)
        end
    end
    split_test = NumericBinaryTest(att_idx, best_suggestion[2])
    return AttributeSplitSuggestion(split_test, best_suggestion[3], best_suggestion[1], 2)
end



