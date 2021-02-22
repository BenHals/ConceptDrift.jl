using Random
abstract type AbstractDataStream end

"""
Information representing a data stream.
"""
mutable struct Stream
    n_samples::Int
    n_targets::Int
    n_features::Int
    n_num_features::Int
    n_cat_features::Int
    n_classes::Int
    n_cat_features_idx::Vector{Int}
    current_sample_x::Vector{Float64}
    current_sample_y::Vector{Int}
    sample_idx::Int
    feature_names::Vector{String}
    target_names::Vector{String}
    target_values::Vector{Int}
    name::String
    random_state_seed::Int
    random_state::AbstractRNG
end

function Stream(;n_samples::Int,
    n_targets::Int,
    n_features::Int,
    n_num_features::Int,
    n_cat_features::Int,
    n_classes::Int,
    n_cat_features_idx::Vector{Int},
    current_sample_x::Vector{Float64},
    current_sample_y::Vector{Int},
    sample_idx::Int,
    feature_names::Vector{String},
    target_names::Vector{String},
    target_values::Vector{Int},
    name::String,
    random_state_seed::Int,
    random_state::AbstractRNG)

    Stream(n_samples, n_targets, n_features, n_num_features, n_cat_features,
        n_classes, n_cat_features_idx, current_sample_x, current_sample_y, sample_idx, 
        feature_names, target_names, target_values, name, random_state_seed, random_state)
end

function Stream(name::String, n_features::Int64)
    seed = 42
    rng = MersenneTwister(seed)
    return Stream(-1, 1, n_features, n_features, 0, 2, [], [], [], 0, ["f$(i)" for i in 1:n_features], ["label"], [0, 1], name, seed, rng)
end

function get_stream_name(s::AbstractDataStream)
    return s.stream.name
end

"""
Generate the next X,y sample in the stream.
X is a 2 dimensional vector, uniformly sampled between 0-1.
y is determined by the classification_function of the SineGenerator.
"""
function next_sample!(s::AbstractDataStream, batch_size)
    data_X = Array{Float64}(undef, s.stream.n_features, batch_size)
    data_Y = Vector{Int64}(undef, batch_size)
    for j in 1:batch_size
        s.stream.sample_idx += 1
        for f in 1:s.stream.n_features
            feature = next_feature(s, f)
            data_X[f, j] = feature
        end
        data_Y[j] = s.classification_function(data_X[:, j]...)
        s.stream.current_sample_x = view(data_X, :, j)
        s.stream.current_sample_y = view(data_Y, j:j)
    end
    return data_X, data_Y
end


"""
Reset the RNG state sample counter to starting state.
"""
function stream_reset!(s::AbstractDataStream)
    rng = MersenneTwister(s.stream.random_state_seed)
    s.stream.random_state = rng
    s.stream.sample_idx = 0
    return
end

