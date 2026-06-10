# Training attention (attention! / ∇attention!) against Float64 references.
# The backward reference is the standard attention gradient: with
# S = KᵀQ·scale (+bias), P = softmax(S), O = V·P and incoming Ō:
#   P̄ = Vᵀ·Ō,  S̄ = P ∘ (P̄ - colsum(P ∘ P̄)),
#   Q̄ = K·S̄·scale,  K̄ = Q·S̄ᵀ·scale,  V̄ = Ō·Pᵀ,  B̄ = S̄ (summed over broadcast).

function attn_ref(Q, K, V, Bias = nothing; causal = false)
    Dk, M, H, B = size(Q)
    Dv, N, Hkv, _ = size(V)
    g = H ÷ Hkv
    scale = 1 / sqrt(Dk)
    O = zeros(Float64, Dv, M, H, B)
    for b in 1:B, h in 1:H
        hk = cld(h, g)
        S = (transpose(K[:, :, hk, b]) * Q[:, :, h, b]) .* scale
        if !isnothing(Bias)
            S .+= Bias[:, :, mod1(h, size(Bias, 3)), mod1(b, size(Bias, 4))]
        end
        if causal
            for m in 1:M, n in 1:N
                n > m && (S[n, m] = -Inf)
            end
        end
        P = exp.(S .- maximum(S; dims = 1))
        P ./= sum(P; dims = 1)
        O[:, :, h, b] = V[:, :, hk, b] * P
    end
    return O
end

function attn_bwd_ref(Ō, Q, K, V, Bias = nothing; causal = false)
    Dk, M, H, B = size(Q)
    Dv, N, Hkv, _ = size(V)
    g = H ÷ Hkv
    scale = 1 / sqrt(Dk)
    Q̄ = zero(Q); K̄ = zero(K); V̄ = zero(V)
    B̄ = isnothing(Bias) ? nothing : zero(Bias)
    for b in 1:B, h in 1:H
        hk = cld(h, g)
        S = (transpose(K[:, :, hk, b]) * Q[:, :, h, b]) .* scale
        if !isnothing(Bias)
            S .+= Bias[:, :, mod1(h, size(Bias, 3)), mod1(b, size(Bias, 4))]
        end
        if causal
            for m in 1:M, n in 1:N
                n > m && (S[n, m] = -Inf)
            end
        end
        P = exp.(S .- maximum(S; dims = 1))
        P ./= sum(P; dims = 1)

        ō = Ō[:, :, h, b]
        P̄ = transpose(V[:, :, hk, b]) * ō
        S̄ = P .* (P̄ .- sum(P .* P̄; dims = 1))
        Q̄[:, :, h, b] = K[:, :, hk, b] * S̄ .* scale
        K̄[:, :, hk, b] .+= Q[:, :, h, b] * transpose(S̄) .* scale
        V̄[:, :, hk, b] .+= ō * transpose(P)
        isnothing(B̄) || (B̄[:, :, mod1(h, size(Bias, 3)), mod1(b, size(Bias, 4))] .+= S̄)
    end
    return Q̄, K̄, V̄, B̄
end

attn_inputs(rng; Dk = 64, Dv = 64, M = 96, N = 128, H = 4, Hkv = H, B = 2) = (
    randn(rng, Float64, Dk, M, H, B),
    randn(rng, Float64, Dk, N, Hkv, B),
    randn(rng, Float64, Dv, N, Hkv, B),
)

const ATTN_ATOL = 1f-2
const ATTN_RTOL = 3f-2

≈ᵦ(A, B) = isapprox(A, B; atol = ATTN_ATOL, rtol = ATTN_RTOL)

@testset "attention!" begin
    rng = MersenneTwister(8)

    @testset "causal=$causal" for causal in (false, true)
        Q, K, V = attn_inputs(rng)
        dQ, dK, dV = map(x -> CuArray(Float32.(x)), (Q, K, V))
        dO = similar(dQ, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
        attention!(dO, dQ, dK, dV; causal)
        @test Array(dO) ≈ᵦ attn_ref(Q, K, V; causal)
    end

    @testset "bias (broadcast over heads/batch)" begin
        Q, K, V = attn_inputs(rng)
        Bias = randn(rng, Float64, size(K, 2), size(Q, 2), 1, 1)
        dQ, dK, dV = map(x -> CuArray(Float32.(x)), (Q, K, V))
        dB = CuArray(Float32.(Bias))
        dO = similar(dQ, size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))
        attention!(dO, dQ, dK, dV, dB)
        @test Array(dO) ≈ᵦ attn_ref(Q, K, V, Bias)
    end
end

@testset "∇attention!" begin
    rng = MersenneTwister(9)

    function run_bwd(Q, K, V, Bias; causal)
        dQ, dK, dV = map(x -> CuArray(Float32.(x)), (Q, K, V))
        dB = isnothing(Bias) ? nothing : CuArray(Float32.(Bias))
        Dv, M, H, B = size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4)
        dO = similar(dQ, Dv, M, H, B)
        dM = similar(dQ, Float32, M, H, B)
        dL = similar(dQ, Float32, M, H, B)
        attention!(dO, dQ, dK, dV, dB; causal, M = dM, L = dL)

        Ō = randn(rng, Float64, Dv, M, H, B)
        dŌ = CuArray(Float32.(Ō))
        dQ̄, dK̄, dV̄ = similar(dQ), similar(dK), similar(dV)
        dB̄ = isnothing(Bias) ? nothing : similar(dB)
        ∇attention!(dQ̄, dK̄, dV̄, dB̄, dŌ, dQ, dK, dV, dB, dO, dM, dL; causal)

        refs = attn_bwd_ref(Ō, Q, K, V, Bias; causal)
        outs = (Array(dQ̄), Array(dK̄), Array(dV̄), isnothing(dB̄) ? nothing : Array(dB̄))
        return outs, refs
    end

    @testset "MHA causal=$causal" for causal in (false, true)
        Q, K, V = attn_inputs(rng)
        (Q̄, K̄, V̄, _), (Q̄r, K̄r, V̄r, _) = run_bwd(Q, K, V, nothing; causal)
        @test Q̄ ≈ᵦ Q̄r
        @test K̄ ≈ᵦ K̄r
        @test V̄ ≈ᵦ V̄r
    end

    @testset "GQA (atomic K̄/V̄ accumulation)" begin
        Q, K, V = attn_inputs(rng; H = 4, Hkv = 2)
        (Q̄, K̄, V̄, _), (Q̄r, K̄r, V̄r, _) = run_bwd(Q, K, V, nothing; causal = true)
        @test Q̄ ≈ᵦ Q̄r
        @test K̄ ≈ᵦ K̄r
        @test V̄ ≈ᵦ V̄r
    end

    @testset "bias grads ($label)" for (label, bh, bb) in
            [("full, plain store", nothing, nothing), ("broadcast, atomic", 1, 1)]
        Q, K, V = attn_inputs(rng)
        H, B = size(Q, 3), size(Q, 4)
        Bias = randn(rng, Float64, size(K, 2), size(Q, 2),
                     something(bh, H), something(bb, B))
        (Q̄, K̄, V̄, B̄), (Q̄r, K̄r, V̄r, B̄r) = run_bwd(Q, K, V, Bias; causal = false)
        @test Q̄ ≈ᵦ Q̄r
        @test K̄ ≈ᵦ K̄r
        @test V̄ ≈ᵦ V̄r
        @test B̄ ≈ᵦ B̄r
    end
end
