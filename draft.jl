include("src\\JuliaStream.jl")

a = JuliaStream.SineGenerator(1)
c = JuliaStream.NaiveBayes(Dict(), Dict(), [])
for i in 1:1000000
    X,y = JuliaStream.next_sample!(a, 1)
    JuliaStream.partial_fit!(c, vec(X), y[1])
end

# a = JuliaStream.SineGenerator(2)
right = 0
wrong = 0
for i in 1:100000
    X,y = JuliaStream.next_sample!(a, 1)
    label = JuliaStream.predict(c, vec(X))
    if label == y[1]
        right += 1
    else
        wrong += 1
    end
end

acc = right / (right + wrong)


o = JuliaStream.NumericAttributeObserver()

JuliaStream.update(o, 0.5, 1)
JuliaStream.update(o, 0.75, 1)
JuliaStream.update(o, 0.75, 0)
JuliaStream.update(o, 1.75, 0)
JuliaStream.update(o, 0.75, 0)
o
JuliaStream.range(o)

JuliaStream.get_split_point_suggestions(o)
JuliaStream.estimated_weight_split(o.distribution_per_class[1], 0.25)

JuliaStream.get_binary_split_class_distribution(o, 1.0)
split = JuliaStream.get_best_split_suggestion(o, Dict(0=>3.0, 1=>2.0), 1)

JuliaStream.branch_for_instance(split.split_test, [0.6, 1.0])

JuliaStream.pdf(o.distribution_per_class[1], 0.625)
stdev = JuliaStream.stdev(o.distribution_per_class[1])
pdf(Normal(o.distribution_per_class[1].mean, stdev), 0.25)

n = JuliaStream.LearningNodeNB()
JuliaStream.learn_one!(n, [0.5], 1)
JuliaStream.learn_one!(n, [0.75], 1)
JuliaStream.learn_one!(n, [0.75], 0)
JuliaStream.learn_one!(n, [1.75], 0)
JuliaStream.learn_one!(n, [0.75], 0)
n

JuliaStream.get_best_split_suggestions(n)

JuliaStream.predict_one(n, [0.5])

h = JuliaStream.HoeffdingTree()
h
leaf = JuliaStream.filter_instance_to_leaf(h.tree_root, [1.1, 0.1], nothing, -1)
JuliaStream.learn_one!(leaf.node, [0.5, 1.1], 0)
JuliaStream.partial_fit!(h, [0.5, 1.1], 0)
JuliaStream.partial_fit!(h, [0.3, 1.1], 0)
JuliaStream.partial_fit!(h, [0.4, 1.2], 0)
JuliaStream.partial_fit!(h, [0.6, 0.8], 0)
JuliaStream.partial_fit!(h, [1.5, 0.1], 1)
JuliaStream.partial_fit!(h, [1.3, 0.1], 1)
JuliaStream.partial_fit!(h, [1.4, 0.2], 1)
JuliaStream.partial_fit!(h, [1.6, 0.3], 1)
JuliaStream.predict(h, [1.4, 0.3])
JuliaStream.predict(h, [0.4, 1.3])

a = JuliaStream.SineGenerator(3)
c = JuliaStream.HoeffdingTree()
for i in 1:1000000
    X,y = JuliaStream.next_sample!(a, 1)
    JuliaStream.partial_fit!(c, vec(X), y[1])
end

# a = JuliaStream.SineGenerator(2)
right = 0
wrong = 0
for i in 1:100000
    X,y = JuliaStream.next_sample!(a, 1)
    label = JuliaStream.predict(c, vec(X))
    if label == y[1]
        right += 1
    else
        wrong += 1
    end
end

acc = right / (right + wrong)
c
leaf = JuliaStream.filter_instance_to_leaf(h.tree_root, [1.1, 0.1], nothing, -1)
n = leaf.node
JuliaStream.get_best_split_suggestions(n)


sort!(best_split_suggestions, by = x -> x.merit)
split_decision = best_split_suggestions[end]
split_decision.resulting_class_distribution
for (i, child_distribution) in enumerate(split_decision.resulting_class_distribution)
    child_distribution
end

JuliaStream.hoeffding_bound(1, 0.000001, 0.3)



abstract type Asset end

abstract type Property <: Asset end
abstract type Investment <: Asset end
abstract type Cash <: Asset end

abstract type House <: Property end
abstract type Apartment <: Property end

abstract type FixedIncome <: Investment end
abstract type Equity <: Investment end
abstract type LiquidityStyle end
struct Residence <: House
    location
 end
 
 struct Stock <: Equity
     symbol
     name
 end
 
 struct TreasuryBill <: FixedIncome
     cusip
 end
 
 struct Money <: Cash
     currency
     amount
 end
struct IsLiquid <: LiquidityStyle end
struct IsIlliquid <: LiquidityStyle end
# Default behavior is illiquid
LiquidityStyle(::Type) = IsIlliquid()

# Cash is always liquid
LiquidityStyle(::Type{<:Cash}) = IsLiquid()

# Any subtype of Investment is liquid
LiquidityStyle(::Type{<:Investment}) = IsLiquid()
tradable(x::T) where {T} = tradable(LiquidityStyle(T), x)
tradable(::IsLiquid, x) = true
tradable(::IsIlliquid, x) = false

s = Stock(:TSLA, "tesla")
function  t(x::T) where {T}
    return LiquidityStyle(T)
end

r = t(s)
typeof(r)


JuliaStream.get_cat_value_from_onehot([0, 0, 1], 1, 0, 3)
