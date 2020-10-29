struct NumericAttributeObserver
    distribution_per_class::Dict{Int, GaussianEstimator}
end

function NumericAttributeObserver()
    return NumericAttributeObserver(Dict())
end

function update(o::NumericAttributeObserver, x::Float64, y::Int64)
    if !haskey(o.distribution_per_class, y)
        o.distribution_per_class[y] = GaussianEstimator()
    end
    o.distribution_per_class[y] = add_weight(o.distribution_per_class[y], x, 1.0)
    return
end