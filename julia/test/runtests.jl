using Metatensor
using Test

TESTS = [
    "labels.jl"
]

function main()
    @testset "Version" begin
        @test startswith(Metatensor.lib.version(), "0.")
    end

    for test in TESTS
        include(test)
    end
end

main()
