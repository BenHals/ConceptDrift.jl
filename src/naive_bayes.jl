struct NaiveBayes
    observed_class_distribution::Dict{Int, Int}
    attribute_observers::Dict{Int, NumericAttributeObserver}
    classes::Vector{Int}
end

function NaiveBayes()
    return NaiveBayes(Dict(), Dict(), [])
end

function partial_fit!(c::NaiveBayes, X::Vector{Float64}, y::Int64)
    if haskey(c.observed_class_distribution, y)
        c.observed_class_distribution[y] += 1
    else
        c.observed_class_distribution[y] = 1
    end
    for i in 1:length(X)
        if !haskey(c.attribute_observers, i)
            c.attribute_observers[i] = NumericAttributeObserver()
        end
        update(c.attribute_observers[i], X[i], y)
    end

end

function argmax_votes(votes::Dict{Int64, Float64})
    maxkey = 0
    maxvalue = 0
    for (key, value) in votes
        if value >= maxvalue
            maxkey = key
            maxvalue = value
        end
    end
    return maxkey
end

function predict_proba(c::NaiveBayes, X::Vector{Float64})
    sum_seen_weights = sum(collect(values(c.observed_class_distribution)))
    votes = Dict(i => c.observed_class_distribution[i]/sum_seen_weights for i in keys(c.observed_class_distribution))
    for feature_idx in keys(c.attribute_observers)
        for class_idx in keys(votes)
            class_likelihood = pdf(c.attribute_observers[feature_idx].distribution_per_class[class_idx], X[feature_idx])
            votes[class_idx] *= class_likelihood
        end
    end
    return votes
end

function predict(c::NaiveBayes, X::Vector{Float64})
    votes = predict_proba(c, X)
    return argmax_votes(votes)
end

