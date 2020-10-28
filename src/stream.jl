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

struct SineGenerator{F<:Function} <: AbstractDataStream
    stream::Stream
    classification_function::F
end

function SineGenerator(name::String)
    seed = 42
    rng = MersenneTwister(seed)
    return SineGenerator(Stream(0, 0, 0, 0, 0, 0, [], [], [], 0, [], [], [], name, seed, rng), sine_func_1)
end

function sine_func_1(a, b)
    return a+b
end

function get_stream_name(s::AbstractDataStream)
    return s.stream.name
end

function next_sample(s::AbstractDataStream, batch_size)
    data = Array{Float64, 2}(undef, batch_size, s.stream.n_features)
    att1 = 0
    att2 = 0
    for j in 1:batch_size
        s.stream.sample_idx += 1
        att1 = rand(s.stream.random_state)
        att2 = rand(s.stream.random_state)
    end
    return att1,att2
end
