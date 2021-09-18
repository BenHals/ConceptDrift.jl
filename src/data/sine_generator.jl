"""
An infinite length 2 feature data stream generator based on the sine function
as described in [1].

# Arguments:
* `stream::Stream`: A container of data stream options
* `classification_function`: One of the 4 functions `sine_func_[0, 1, 2, 3]`.
    Determines how the label is generated from features.

[1] Gama, Joao, et al.'s 'Learning with drift
detection.' Advances in artificial intelligence–SBIA 2004. Springer Berlin
Heidelberg, 2004. 286-295."

"""
struct SineGenerator{F<:Function} <: AbstractDataStream
    stream::Stream
    classification_function::F
    classification_function_idx::Int
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

function select_sine_func(classification_function_idx::Int64)
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

function classify_instance(s::SineGenerator, X)
    s.classification_function(X...)
end

"""
Initialize a SineGenerator using an Int to select a classification_function.
"""
function SineGenerator(classification_function_idx::Int64)
    classification_function = select_sine_func(classification_function_idx)
    return SineGenerator(Stream("SineGenerator", 2), classification_function, classification_function_idx)
end

"""
Each feature is uniform random in range [0, 1]
"""
function next_num_feature(s::SineGenerator, f_id::Int)
    rand(s.stream.random_state)
end


