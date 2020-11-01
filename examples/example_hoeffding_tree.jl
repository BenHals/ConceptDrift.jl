include("src\\JuliaStream.jl")

function get_classifier_accuracy_test(c::JuliaStream.HoeffdingTree, a::JuliaStream.AbstractDataStream)
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
end

a = JuliaStream.SineGenerator(0)
c = JuliaStream.HoeffdingTree()
test_result = @timed get_classifier_accuracy_test(c, a)