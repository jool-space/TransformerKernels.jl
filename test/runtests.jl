using TransformerKernels
using CUDA
using LinearAlgebra
using Random
using Test

@testset "TransformerKernels.jl" begin
    @test CUDA.functional()

    include("decode.jl")
    include("softmax.jl")
    include("norm.jl")
end
