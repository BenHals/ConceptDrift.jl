abstract type AbstractNode end

struct LearningNodeNB <: AbstractNode
    stats::Dict{Int64, Int64}
    classifier::NaiveBayes
    last_split_attempt::Int64
end

function LearningNodeNB()
    return LearningNodeNB(Dict(), NaiveBayes(), 0)
end

struct SplitNode <: AbstractNode
    stats::Dict{Int64, Int64}
    children::Dict{Int64, AbstractNode}
end

function SplitNode()
    return SplitNode(Dict(), Dict())
end


struct HoeffdingTree
    grace_period::Int64
    split_confidence::Float64
    tree_root::AbstractNode
    decision_node_count::Int64
    active_leaf_node_count::Int64
    train_weight_seen::Float64
end

# function HoeffdingTree()
#     return HoeffdingTree(200, 0.0000001, )