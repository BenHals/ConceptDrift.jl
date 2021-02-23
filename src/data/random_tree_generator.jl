"""
An infinite length data stream generator based on a random decision tree

# Arguments:
* `stream::Stream`: A container of data stream options
* `classification_function`: One of the 4 functions `sine_func_[0, 1, 2, 3]`.
    Determines how the label is generated from features.

"""
@enum NodeType begin
    numeric
    categoric
end
struct RTreeNode
    class_label::Union{Int, Missing}
    type::NodeType
    split_att_index::Int
    split_att_value::Float64
    children::Array{RTreeNode}
end

struct RandomTreeGenerator <: AbstractDataStream
    stream::Stream
    tree_random_seed::Int
    tree_random_state::AbstractRNG
    n_categories_per_cat_feature::Int
    max_leaf_depth::Int
    min_leaf_depth::Int
    fraction_leaves_per_level::Float64
    tree_root::RTreeNode
    cat_var_selector::Base.RefValue{Int}
end

"""
Initialize a RandomTreeGenerator using an Int to select a classification_function.
"""
function RandomTreeGenerator(;tree_random_state::Int=-1,
                        sample_random_state::Int=-1,
                        n_classes::Int=2,
                        n_cat_features=5,
                        n_num_features=5,
                        n_categories_per_cat_feature=5,
                        max_leaf_depth=5,
                        min_leaf_depth=3,
                        fraction_leaves_per_level=0.15
                        )
    if tree_random_state == -1
        tree_random_state = Random.rand(1:1000)
    end
    if sample_random_state == -1
        sample_random_state = Random.rand(1:1000)
    end
    rng = MersenneTwister(sample_random_state)
    tree_rng = MersenneTwister(tree_random_state)

    return RandomTreeGenerator(Stream(n_samples=-1, 
                                    n_targets=1,
                                    n_features=(n_cat_features * n_categories_per_cat_feature)+n_num_features,
                                    n_num_features=n_num_features,
                                    n_cat_features=n_cat_features,
                                    n_classes=n_classes,
                                    n_cat_features_idx=[i for i in n_num_features+1:n_num_features + (n_cat_features * n_categories_per_cat_feature)],
                                    current_sample_x=Array{Float64, 1}(),
                                    current_sample_y=Array{Int, 1}(),
                                    sample_idx=0,
                                    feature_names=vcat(["f$(i)" for i in 1:n_num_features], ["c$(i)$(v)" for i in n_num_features+1:n_num_features + n_cat_features for v in 1:n_categories_per_cat_feature]),
                                    target_names=["label"],
                                    target_values=[0, 1],
                                    name="RandomTreeGenerator",
                                    random_state_seed=sample_random_state,
                                    random_state=rng),
                                    tree_random_state,
                                    tree_rng,
                                    n_categories_per_cat_feature,
                                    max_leaf_depth,
                                    min_leaf_depth,
                                    fraction_leaves_per_level,
                                    _generate_random_tree(
                                        tree_rng,
                                        n_classes,
                                        n_cat_features,
                                        n_num_features,
                                        n_categories_per_cat_feature,
                                        max_leaf_depth,
                                        min_leaf_depth,
                                    fraction_leaves_per_level),
                                        Ref(0)
                                )
end

function _generate_random_tree(tree_rng::AbstractRNG,
                                n_classes::Int,
                                n_cat_features::Int,
                                n_num_features::Int,
                                n_categories_per_cat_feature::Int,
                                max_leaf_depth::Int,
                                min_leaf_depth::Int,
                                fraction_leaves_per_level::Float64)
    _generate_random_tree_node(0, [i for i in 1:n_num_features+n_cat_features], [0.0 for i in 1:n_num_features], [1.0 for i in 1:n_num_features],
                            tree_rng, n_classes, n_cat_features, n_num_features, n_categories_per_cat_feature, max_leaf_depth, min_leaf_depth, fraction_leaves_per_level)
end

function  _generate_random_tree_node(current_depth::Int,
                                    nominal_candidates::Array{Int},
                                    min_numeric_values::Array{Float64},
                                    max_numeric_values::Array{Float64},
                                    tree_rng::AbstractRNG,
                                    n_classes::Int,
                                    n_cat_features::Int,
                                    n_num_features::Int,
                                    n_categories_per_cat_feature::Int,
                                    max_leaf_depth::Int,
                                    min_leaf_depth::Int,
                                    fraction_leaves_per_level::Float64)
    leaf_rng = rand(tree_rng) > fraction_leaves_per_level
    set_leaf = (current_depth >= max_leaf_depth) || (current_depth >= min_leaf_depth && leaf_rng)
    if set_leaf
        class_label = rand(tree_rng, 1:n_classes)
        leaf = RTreeNode(class_label, numeric, -1, -1, [])
        return leaf
    end
    
    split_attribute = rand(tree_rng, 1:length(nominal_candidates))
    if split_attribute <= n_num_features
        min_value = min_numeric_values[split_attribute]
        max_value = max_numeric_values[split_attribute]
        split_point = min_value + (max_value - min_value) * rand(tree_rng)
        left_child = _generate_random_tree_node(
            current_depth + 1,
            nominal_candidates,
            min_numeric_values,
            [max_numeric_values[1:split_attribute]..., split_point, max_numeric_values[split_attribute+2:length(max_numeric_values)]...],
            tree_rng,
            n_classes,
            n_cat_features,
            n_num_features,
            n_categories_per_cat_feature,
            max_leaf_depth,
            min_leaf_depth,
            fraction_leaves_per_level
        )
        right_child = _generate_random_tree_node(
            current_depth + 1,
            nominal_candidates,
            [min_numeric_values[1:split_attribute]..., split_point, min_numeric_values[split_attribute+2:length(min_numeric_values)]...],
            max_numeric_values,
            tree_rng,
            n_classes,
            n_cat_features,
            n_num_features,
            n_categories_per_cat_feature,
            max_leaf_depth,
            min_leaf_depth,
            fraction_leaves_per_level
        )
        return RTreeNode(missing, numeric, split_attribute, split_point, [left_child, right_child])
    else
        new_nominal_candidates = [nominal_candidates[1:split_attribute]..., nominal_candidates[split_attribute+2:length(nominal_candidates)]...]
        children = [_generate_random_tree_node(
            current_depth + 1,
            new_nominal_candidates,
            min_numeric_values,
            max_numeric_values,
            tree_rng,
            n_classes,
            n_cat_features,
            n_num_features,
            n_categories_per_cat_feature,
            max_leaf_depth,
            min_leaf_depth,
            fraction_leaves_per_level) for i in 1:n_categories_per_cat_feature]
            return RTreeNode(missing, categoric, split_attribute, 0, children)
    end
end

function _classify_instance(node, X, n_num_features, n_categories_per_cat_feature)
    if !ismissing(node.class_label)
        return node.class_label
    end
    attribute_value = X[node.split_att_index]
    if node.type == numeric
        child_id = attribute_value < node.split_att_value ? 1 : 2
        return _classify_instance(node.children[child_id], X, n_num_features, n_categories_per_cat_feature)
    else
        child_id = get_cat_value_from_onehot(X, node.split_att_index, n_num_features, n_categories_per_cat_feature)
        return _classify_instance(node.children[child_id], X, n_num_features, n_categories_per_cat_feature)
    end
end

"""
Each input vector X is laid out [N1, N2, ... C1_A1, C1_A2...]
I.e all numeric attributes are first then all categoric.
Each categoric attribute is one hot encoded, so covers
indexes [i, i+num_levels] in X.
We scan this range for a 1 to find the true value.
"""
function get_cat_value_from_onehot(X, attribute_index, n_num_features, n_categories_per_cat_feature)
    num_prev_cat_attrs = attribute_index - n_num_features - 1
    starting_index = (n_num_features + 1) + num_prev_cat_attrs * n_categories_per_cat_feature
    for i in starting_index:starting_index+n_categories_per_cat_feature - 1
        if X[i] == 1.0
            return 1 + (i - starting_index)
        end
    end
end

function classify_instance(s::RandomTreeGenerator, X)
    _classify_instance(s.tree_root, X, s.stream.n_num_features, s.n_categories_per_cat_feature)
end

"""
Each feature is uniform random in range [0, 1]
"""
function next_num_feature(s::RandomTreeGenerator, f_id::Int)
    rand(s.stream.random_state)
end

"""
Return 0 or 1 to build a random one hot encoded element for 
each categoric variable.
"""
function next_cat_feature(s::RandomTreeGenerator, f_id::Int)
    cat_id = f_id - s.stream.n_num_features
    first_elem = f_id % s.n_categories_per_cat_feature == 1
    if first_elem
        s.cat_var_selector[] = rand(s.stream.random_state, 1:s.n_categories_per_cat_feature)
    end
    if cat_id % s.n_categories_per_cat_feature == s.cat_var_selector[] - 1
        return 1.0
    end
    return 0.0
end

