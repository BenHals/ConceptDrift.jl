include("src\\ConceptDrift.jl")

function get_classifier_accuracy_test(c::ConceptDrift.HoeffdingTree, a::ConceptDrift.AbstractDataStream)
    for i in 1:1000000
        X,y = ConceptDrift.next_sample!(a, 1)
        ConceptDrift.partial_fit!(c, vec(X), y[1])
    end

    right = 0
    wrong = 0
    for i in 1:100000
        X,y = ConceptDrift.next_sample!(a, 1)
        label = ConceptDrift.predict(c, vec(X))
        if label == y[1]
            right += 1
        else
            wrong += 1
        end
    end

    acc = right / (right + wrong)
end

# a = ConceptDrift.SineGenerator(0)
a = ConceptDrift.RandomTreeGenerator(min_leaf_depth=5)
c = ConceptDrift.HoeffdingTree()
test_result = @timed get_classifier_accuracy_test(c, a)