
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
    return test_value <= t.split_value ? 1 : 2
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

mutable struct LearningNodeNB{T<:AbstractSplitCritereon} <: AbstractNode
    stats::Dict{Int64, Float64}
    classifier::NaiveBayes
    last_split_attempt::Int64
    split_critereon::T
end

function LearningNodeNB()
    return LearningNodeNB(Dict{Int64, Float64}(), NaiveBayes(), 0, GiniSplitCritereon())
end

function total_weight(n::AbstractNode)
    return sum(values(n.stats))
end

function filter_instance_to_leaf(n::LearningNodeNB, X::Vector{Float64}, parent, parent_branch::Int64)
    return FoundNode(n, parent, parent_branch)
end

function learn_one!(n::LearningNodeNB, X::Vector{Float64}, y::Int64, weight::Float64=1.0)
    update_stats(n, y, weight)
    partial_fit!(n.classifier, X, y, weight)
end

function predict_one(n::LearningNodeNB, X::Vector{Float64})
    return predict_proba(n.classifier, X)
end

function update_stats(n::LearningNodeNB, y::Int64, weight::Float64=1.0)
    if !haskey(n.stats, y)
        n.stats[y] = 0.0
    end
    n.stats[y] += weight
end

function get_best_split_suggestions(n::LearningNodeNB)
    best_suggestions = Vector{AttributeSplitSuggestion}()
    for (attr_idx, observer) in pairs(n.classifier.attribute_observers)
        best_suggestion = get_best_split_suggestion(observer, n.classifier.observed_class_distribution, attr_idx)
        push!(best_suggestions, best_suggestion)
    end
    return best_suggestions
end



struct SplitNode{T<:AbstractSplitCritereon} <: AbstractNode
    stats::Dict{Int64, Float64}
    children::Dict{Int64, AbstractNode}
    split_test::AbstractConditionalTest
    split_critereon::T
end

function set_child(parent::SplitNode, child::AbstractNode, branch_idx)
    parent.children[branch_idx] = child
end

function filter_instance_to_leaf(n::SplitNode, X::Vector{Float64}, parent, parent_branch::Int64)
    child_branch = branch_for_instance(n.split_test, X)
    if child_branch == -1
        return FoundNode(n, parent, parent_branch)
    end
    child = n.children[child_branch]
    return filter_instance_to_leaf(child, X, n, child_branch)
end




mutable struct HoeffdingTree
    grace_period::Float64
    split_confidence::Float64
    tie_threshhold::Float64
    tree_root::AbstractNode
    decision_node_count::Int64
    active_leaf_node_count::Int64
    train_weight_seen::Float64
end

function HoeffdingTree()
    return HoeffdingTree(200.0, 0.0000001, 0.05, LearningNodeNB(), 1, 1, 0.0)
end

function partial_fit!(c::HoeffdingTree, X::Vector{Float64}, y::Int64)
    found_node  = filter_instance_to_leaf(c.tree_root, X, nothing, -1)
    leaf = found_node.node
    handle_leaf(leaf, found_node.parent, found_node.parent_branch, c, X, y)
end

function handle_leaf(learning_node::LearningNodeNB, parent::Union{AbstractNode, Nothing}, parent_branch::Int64, c::HoeffdingTree, X::Vector{Float64}, y::Int64)
    learn_one!(learning_node, X, y)
    weight_seen = total_weight(learning_node)
    seen_since_last_split = weight_seen - learning_node.last_split_attempt
    if seen_since_last_split >= c.grace_period
        attempt_to_split(c, learning_node, parent, parent_branch)
        learning_node.last_split_attempt = floor(Int, weight_seen)
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

function hoeffding_bound(range, confidence, n)
    return sqrt((range^2 * log(1.0/confidence)) / (2.0*n))
end

function attempt_to_split(c::HoeffdingTree, n::LearningNodeNB, parent::Union{SplitNode, Nothing}, parent_idx::Int64)
    best_split_suggestions = get_best_split_suggestions(n)
    sort!(best_split_suggestions, by = x -> x.merit)
    should_split = false
    if length(best_split_suggestions) < 2
        should_split = length(best_split_suggestions) > 1
        
    # end
    else
        hoeffding_value = hoeffding_bound(get_range_of_merit(n.split_critereon), c.split_confidence, total_weight(n))
        best_suggestion = best_split_suggestions[end]
        second_best_suggestion = best_split_suggestions[end-1]
        if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_value) || (hoeffding_value < c.tie_threshhold)
            should_split = true
        end
    end
    if should_split
        split_decision = best_split_suggestions[end]
        split_children = Dict{Int64, AbstractNode}()
        for (i, child_distribution) in enumerate(split_decision.resulting_class_distribution)
            new_child = LearningNodeNB(child_distribution, NaiveBayes(), 0, n.split_critereon)
            split_children[i] = new_child
        end
        new_split = SplitNode(n.stats, split_children, split_decision.split_test, n.split_critereon)
        if !isa(parent, Nothing)
            set_child(parent, new_split, parent_idx)
        else
            c.tree_root = new_split
        end
    end

end

