"""
An infinite length 2 feature data stream generator based on the sine function.

    # Arguments:
    * `stream::Stream`: A container of data stream options
    * `classification_function`: One of the 4 functions `sine_func_[0, 1, 2, 3]`.
            Determines how the label is generated from features.

"""
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

"""
Initialize a SineGenerator using an Int to select a classification_function.
"""
function SineGenerator(classification_function_idx::Int64)
    classification_function = select_sine_func(classification_function_idx)
    return SineGenerator(Stream("SineGenerator", 2), classification_function)
end

"""
Generate the next X,y sample in the stream.
X is a 2 dimensional vector, uniformly sampled between 0-1.
y is determined by the classification_function of the SineGenerator.
"""
function next_sample!(s::SineGenerator, batch_size)
    data_X = Array{Float64}(undef, s.stream.n_features, batch_size)
    data_Y = Vector{Int64}(undef, batch_size)
    for j in 1:batch_size
        s.stream.sample_idx += 1
        for f in 1:s.stream.n_features
            feature = rand(s.stream.random_state)
            data_X[f, j] = feature
        end
        data_Y[j] = s.classification_function(data_X[:, j]...)
        s.stream.current_sample_x = view(data_X, :, j)
        s.stream.current_sample_y = view(data_Y, j:j)
    end
    return data_X, data_Y
end
