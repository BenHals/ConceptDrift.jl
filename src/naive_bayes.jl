struct NaiveBayes
    observed_class_distribution::Dict{Int, Int}
    attribute_observers::Dict{Int, Int}
    classes::Vector{Int}
end

function partial_fit!(c::NaiveBayes, X::Vector{Float64}, y::Int64)
    if haskey(c.observed_class_distribution)
        c.observed_class_distribution[y] += 1
    else
        c.observed_class_distribution[y] = 1
    end
end

    
