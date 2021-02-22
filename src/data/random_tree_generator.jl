"""
An infinite length data stream generator based on a random decision tree

# Arguments:
* `stream::Stream`: A container of data stream options
* `classification_function`: One of the 4 functions `sine_func_[0, 1, 2, 3]`.
    Determines how the label is generated from features.

"""

struct Node
    class_label::Int
    split_att_index::Int
    split_att_value::Int
    children::Array{Node}
end

struct RandomTreeGenerator <: AbstractDataStream
    stream::Stream
    tree_random_seed::Int
    tree_random_state::AbstractRNG
    n_categories_per_cat_feature::Int
    max_tree_depth::Int
    min_leaf_depth::Int
    fraction_leaves_per_level::Float64
    tree_root::Node
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
                        max_tree_depth=5,
                        min_leaf_depth=3,
                        fraction_leaves_per_level=0.15
                        )
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
                                    max_tree_depth,
                                    min_leaf_depth,
                                    fraction_leaves_per_level,
                                    _generate_random_tree(
                                        tree_random_state,
                                        sample_random_state,
                                        n_classes,
                                        n_cat_features,
                                        n_num_features,
                                        n_categories_per_cat_feature,
                                        max_tree_depth,
                                        min_leaf_depth,
                                        fraction_leaves_per_level
                                        ),
                                    )
end

function _generate_random_tree(tree_random_state::Int,
                                    sample_random_state::Int,
                                    n_classes::Int,
                                    n_cat_features,
                                    n_num_features,
                                    n_categories_per_cat_feature,
                                    max_tree_depth,
                                    min_leaf_depth,
                                    fraction_leaves_per_level)
    Node(1, 1, 1, [])
end


"""
Each feature is uniform random in range [0, 1]
"""
function next_feature(s::RandomTreeGenerator, f_id::Int)
    rand(s.stream.random_state)
end


