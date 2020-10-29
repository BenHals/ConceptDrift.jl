include("src\\JuliaStream.jl")

a = JuliaStream.SineGenerator(1)
labels = Int[]
for i in 1:100000
    X,y = JuliaStream.next_sample!(a, 1)
    push!(labels, y)
end
sum(labels)