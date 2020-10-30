using Random
abstract type AbstractDataStream end

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

function Stream(name::String, n_features::Int64)
    seed = 42
    rng = MersenneTwister(seed)
    return Stream(-1, 1, n_features, n_features, 0, 2, [], [], [], 0, ["f$(i)" for i in 1:n_features], ["label"], [0, 1], name, seed, rng)
end

function get_stream_name(s::AbstractDataStream)
    return s.stream.name
end

function stream_reset!(s::AbstractDataStream)
    rng = MersenneTwister(s.stream.random_state_seed)
    s.stream.random_state = rng
    s.stream.sample_idx = 0
    return
end

