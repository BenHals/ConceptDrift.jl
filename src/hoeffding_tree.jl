
abstract type AbstractConditionalTest end

struct NumericBinaryTest <: AbstractConditionalTest
    attr_idx::Int64
    split_value::Float64
end


"""
Return the branch index (index of child) based
on splitting the feature on value given in the test t.

Returns -1 if branch is not found.
"""
function branch_for_instance(t::NumericBinaryTest, X::Vector{Float64})
    if t.attr_idx < 1 || t.attr_idx > length(X)
        return -1
    end
    test_value = X[t.attr_idx]
    return test_value <= t.split_value ? 0 : 1
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
    return lhs_dist, rhs_dist
end

function get_best_split_suggestion(o::NumericAttributeObserver, pre_split_distribution::Dict{Int64, Float64}, att_idx::Int64)
    best_suggestion = (typemin(Float64), 0.0, ())
    suggested_split_values = get_split_point_suggestions(o)
    for split_value in suggested_split_values
        post_split_dist = get_binary_split_class_distribution(o, split_value)
        merit = o.get_split_merit(pre_split_distribution, post_split_dist)
        if merit > best_suggestion[1]
            best_suggestion = (merit, split_value, post_split_dist)
        end
    end
    split_test = NumericBinaryTest(att_idx, best_suggestion[2])
    return AttributeSplitSuggestion(split_test, best_suggestion[3], best_suggestion[1])
end

"""
Calculate the gini impurity of a split as the weighted sum
of the gini impurity values of each child node.
"""
function get_gini_split_merit(pre_split_distribution, post_split_distribution)
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

struct AttributeSplitSuggestion
    split_test::AbstractConditionalTest
    resulting_class_distribution::Tuple{Dict{Int64, Float64}, Dict{Int64, Float64}}
    merit::Float64
end

abstract type AbstractNode end

struct FoundNode
    node::AbstractNode
    parent::Union{AbstractNode, Nothing}
    parent_branch::Int64
    depth::Int64
end

function FoundNode(node, parent, parent_branch)
    return FoundNode(node, parent, parent_branch, -1)
end

mutable struct LearningNodeNB <: AbstractNode
    stats::Dict{Int64, Int64}
    classifier::NaiveBayes
    last_split_attempt::Int64
end

function LearningNodeNB()
    return LearningNodeNB(Dict(), NaiveBayes(), 0)
end

function filter_instance_to_leaf(n::LearningNodeNB, X::Vector{Float64}, parent, parent_branch::Int64)
    return FoundNode(n, parent, parent_branch)
end

function learn_one!(n::LearningNodeNB, X::Vector{Float64}, y::Int64)
    update_stats(n, y)
    partial_fit!(n.classifier, X, y)
end

function predict_one(n::LearningNodeNB, X::Vector{Float64})
    return predict_proba(n.classifier, X)
end

function update_stats(n::LearningNodeNB, y::Int64)
    if !haskey(n.stats, y)
        n.stats[y] = 0
    end
    n.stats[y] += 1
end

function get_best_split_suggestions(n::LearningNodeNB)
    best_suggestions = Vector{AttributeSplitSuggestion}()
    for (attr_idx, observer) in pairs(n.classifier.attribute_observers)
        best_suggestion = get_best_split_suggestion(observer, n.classifier.observed_class_distribution, attr_idx)
        push!(best_suggestions, best_suggestion)
    end
    return best_suggestions
end



struct SplitNode <: AbstractNode
    stats::Dict{Int64, Int64}
    children::Dict{Int64, AbstractNode}
    split_test::AbstractConditionalTest
end




struct HoeffdingTree
    grace_period::Int64
    split_confidence::Float64
    tree_root::AbstractNode
    decision_node_count::Int64
    active_leaf_node_count::Int64
    train_weight_seen::Float64
end

function HoeffdingTree()
    return HoeffdingTree(200, 0.0000001, LearningNodeNB(), 1, 1, 0.0)
end

function partial_fit!(c::HoeffdingTree, X::Vector{Float64}, y::Int64)
    found_node  = filter_instance_to_leaf(c.tree_root, X, nothing, -1)
    leaf = found_node.node
    
    if typeof(leaf) == LearningNodeNB
        learning_node = leaf
        learn_one!(learning_node, X, y)
    end
end

function predict(c::HoeffdingTree, X::Vector{Float64})
    votes = predict_proba(c, X)
    return argmax_votes(votes)
end

function predict_proba(c::HoeffdingTree, X::Vector{Float64})
    found_node  = filter_instance_to_leaf(c.tree_root, X, nothing, -1)
    leaf = found_node.node
    return predict_one(leaf, X)
end