include("src\\JuliaStream.jl")

a = JuliaStream.SineGenerator(1)
c = JuliaStream.NaiveBayes(Dict(), Dict(), [])
for i in 1:1000000
    X,y = JuliaStream.next_sample!(a, 1)
    JuliaStream.partial_fit!(c, vec(X), y[1])
end

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

JuliaStream.pdf(o.distribution_per_class[1], 0.625)
stdev = JuliaStream.stdev(o.distribution_per_class[1])
pdf(Normal(o.distribution_per_class[1].mean, stdev), 0.25)