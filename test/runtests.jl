using TransformerKernels
using Adapt: adapt
using CUDA
using LinearAlgebra
using Random
using Test

@testset "TransformerKernels.jl" begin
    @test CUDA.functional()

    include("flex.jl")
    include("attention.jl")
    include("decode.jl")
    include("softmax.jl")
    include("norm.jl")
end
