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

struct SineGenerator{F<:Function} <: AbstractDataStream
    stream::Stream
    classification_function::F
end

function sine_func_0(a, b)
    return a >= sin(b) ? 0 : 1
end
function sine_func_1(a, b)
    return a < sin(b) ? 0 : 1
end
function sine_func_2(a, b)
    return a >= 0.5 + 0.3 * sin(3 * pi * b) ? 0 : 1
end
function sine_func_3(a, b)
    return a < 0.5 + 0.3 * sin(3 * pi * b) ? 0 : 1
end

function select_sine_func(function_idx::Int64)
    classification_function = sine_func_0
    if classification_function_idx == 1
        classification_function = sine_func_1
    end
    if classification_function_idx == 2
        classification_function = sine_func_2
    end
    if classification_function_idx == 3
        classification_function = sine_func_3
    end
    return classification_function
end

function SineGenerator(classification_function_idx::Int64)
    classification_function = select_sine_func(classification_function_idx)
    return SineGenerator(Stream("SineGenerator", 2), classification_function)
end


function next_sample!(s::SineGenerator, batch_size)
    data_X = Array{Float64, 2}(undef, batch_size, s.stream.n_features)
    data_Y = Vector{Int64}(undef, batch_size)
    for j in 1:batch_size
        s.stream.sample_idx += 1
        for f in 1:s.stream.n_features
            feature = rand(s.stream.random_state)
            data_X[j, f] = feature
        end
        data_Y[j] = s.classification_function(data_X[j, :]...)
        s.stream.current_sample_x = view(data_X, j, :)
        s.stream.current_sample_y = view(data_Y, j:j)
    end
    return data_X, data_Y
end