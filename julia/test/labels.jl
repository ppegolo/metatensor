@testset "Labels" begin
    labels = Labels(["a", "b"], [[2, 3] [4, 5] [6, 7]])

    @test size(labels) == (2, 3)
    @test names(labels) == ["a", "b"]
    @test values(labels) == [[2, 3] [4, 5] [6, 7]]
end
