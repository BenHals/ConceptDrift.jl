using JuliaStream
using Test
using Statistics

@testset "JuliaStream.jl" begin
    # Write your tests here.
end


@testset "gaussian_estimator.jl" begin
    g = JuliaStream.GaussianEstimator()
    @test g.weight_sum == 0.0
    @test g.mean == 0.0
    @test g.variance_sum == 0.0
    test_stream_A = [1.0, 1.0, 2.0]
    for (i, value) in enumerate(test_stream_A)
        g = JuliaStream.add_weight(g, value, 1.0)
        @test g.weight_sum == i
        @test g.mean == mean(test_stream_A[1:i])
        if i > 1
            @test isapprox(JuliaStream.variance(g), var(test_stream_A[1:i]))
            @test isapprox(JuliaStream.stdev(g), std(test_stream_A[1:i]))
        end
    end
    g = JuliaStream.GaussianEstimator()
    @test g.weight_sum == 0.0
    @test g.mean == 0.0
    @test g.variance_sum == 0.0
    test_stream_unweighted = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 1.0, 1.0]
    test_stream_weighted = [1.0, 1.0, 1.0, 3.0, 4.0, 1.0]
    for (i, value) in enumerate(test_stream_weighted)
        g = JuliaStream.add_weight(g, value, 2.0)
        @test g.weight_sum == i*2
        @test g.mean == mean(test_stream_unweighted[1:i*2])
        if i > 1
            @test isapprox(JuliaStream.variance(g), var(test_stream_unweighted[1:i*2]))
            @test isapprox(JuliaStream.stdev(g), std(test_stream_unweighted[1:i*2]))
        end
    end
    g = JuliaStream.GaussianEstimator()
    @test g.weight_sum == 0.0
    @test g.mean == 0.0
    @test g.variance_sum == 0.0
    test_stream_neg = [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 3.0, 3.0, -4.0, 4.0, -1.0, 1.0]
    for (i, value) in enumerate(test_stream_neg)
        g = JuliaStream.add_weight(g, value, 1.0)
        @test g.weight_sum == i
        @test isapprox(g.mean, mean(test_stream_neg[1:i]))
        if i > 1
            @test isapprox(JuliaStream.variance(g), var(test_stream_neg[1:i]))
            @test isapprox(JuliaStream.stdev(g), std(test_stream_neg[1:i]))
        end
    end

end