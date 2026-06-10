export flex_attention!
export ∇flex_attention!

# FlexAttention — variant-agnostic forward kernel; the variant (score_mod /
# mask_mod, see mods.jl) is chosen by TYPE, so each combination compiles to a
# specialized kernel with the mod inlined.
#
# Layout (column-major), matching `attention!`:
#   Q (Dk, SeqLen_Q, Heads, Batch)
#   K (Dk, SeqLen_K, Heads_KV, Batch)
#   V (Dv, SeqLen_K, Heads_KV, Batch)
#   O (Dv, SeqLen_Q, Heads, Batch)
#
# Two SEPARATE single-loop kernels (analytic range vs BlockMask walk): a
# bisection showed no individual heavy construct is at fault, but co-locating
# BOTH loop bodies in one kernel tips a cuTile codegen/structurizer threshold
# (nondeterministic miscompiles). Each kernel carries the body once.

# In-kernel analytic block range + full/partial split (mods.jl lattice).
function flex_fwd(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::Optional{TileArray3{Float32}}, L::Optional{TileArray3{Float32}},
    score_mod, mask_mod,
    qk_scale::Float32, input_pos::Int32,
    H::Int,
    Tc::Type, Tacc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    EVEN_K::Bool, BLOCK_SPARSE::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    # 0-based positions, as lazy descriptors. Mods broadcast through
    # Broadcast.broadcastable; BiasScore reads block_idx for ct.load.
    q_pos = BlockPos{2, TILE_M}(i - 1i32, input_pos)

    m_i = fill(-Inf32, (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Tacc, (Dv, TILE_M))

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)

    q_len = size(Q, 2)
    k_len = size(K, 2)
    num_kv = cld(k_len, Int32(TILE_N))
    iq = i - 1i32                             # 0-based query block
    TMi = Int32(TILE_M); TNi = Int32(TILE_N)

    if BLOCK_SPARSE
        lo, hi = kv_block_range(mask_mod, iq, TMi, TNi, q_len, k_len, input_pos)
        lo = max(lo, 0i32)
        hi = min(hi, num_kv)
    else
        lo = 0i32; hi = num_kv
    end

    for j in lo:hi-1i32
        k = ct.load(K, (1, j + 1i32, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, zeros(Tacc, (TILE_N, TILE_M)))
        s = s * Tacc(qk_scale)                # mods see true scores S = QKᵀ/√dₖ

        kv_pos = BlockPos{1, TILE_N}(j)
        s = score_mod(s, b, h, q_pos, kv_pos)

        if BLOCK_SPARSE && kv_block_full(mask_mod, iq, j, TMi, TNi, q_len, k_len, input_pos)
            # provably no element masked ⇒ skip mask_mod (score_mod only).
            if !EVEN_K
                s = ifelse.(kv_pos .< k_len, s, Tacc(-Inf32))
            end
        else
            umask = mask_mod(b, h, q_pos, kv_pos)
            if !EVEN_K
                umask = umask .& (kv_pos .< k_len)
            end
            s = ifelse.(umask, s, Tacc(-Inf32))
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        safe = m_ij .> -Inf32                 # fully-masked column ⇒ keep -Inf state
        p = ifelse.(safe, exp.(s .- m_ij), 0f0)
        l_ij = sum(p, dims=1)
        alpha = ifelse.(safe, exp.(m_i .- m_ij), 1f0)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* Tacc.(alpha)

        v = ct.load(V, (1, j + 1i32, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)
    ct.store(O, (1, i, h, b), o → eltype(O))
    isnothing(M) || ct.store(M, (i, h, b), reshape(m_i, TILE_M))
    isnothing(L) || ct.store(L, (i, h, b), reshape(l_i, TILE_M))

    return
end

# Walk a precomputed coarse BlockMask index list (mods.jl `build_block_mask`).
function flex_fwd_bm(
    Q::TileArray4, K::TileArray4, V::TileArray4, O::TileArray4,
    M::Optional{TileArray3{Float32}}, L::Optional{TileArray3{Float32}},
    bm_count::TileVector{Int32},
    bm_idx::TileMatrix{Int32},
    bm_full::TileMatrix{Int32},
    score_mod, mask_mod,
    qk_scale::Float32, input_pos::Int32,
    H::Int,
    Tc::Type, Tacc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    EVEN_K::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    q_pos = BlockPos{2, TILE_M}(i - 1i32, input_pos)

    m_i = fill(-Inf32, (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Tacc, (Dv, TILE_M))

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)
    k_len = size(K, 2)

    nb = bm_count[i]
    for t in 1i32:nb
        j = bm_idx[t, i]                      # 0-based KV block
        is_full = bm_full[t, i] != 0i32

        k = ct.load(K, (1, j + 1i32, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, zeros(Tacc, (TILE_N, TILE_M)))
        s = s * Tacc(qk_scale)

        kv_pos = BlockPos{1, TILE_N}(j)
        s = score_mod(s, b, h, q_pos, kv_pos)

        if is_full
            if !EVEN_K
                s = ifelse.(kv_pos .< k_len, s, Tacc(-Inf32))
            end
        else
            umask = mask_mod(b, h, q_pos, kv_pos)
            if !EVEN_K
                umask = umask .& (kv_pos .< k_len)
            end
            s = ifelse.(umask, s, Tacc(-Inf32))
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        safe = m_ij .> -Inf32
        p = ifelse.(safe, exp.(s .- m_ij), 0f0)
        l_ij = sum(p, dims=1)
        alpha = ifelse.(safe, exp.(m_i .- m_ij), 1f0)
        l_i = l_i .* alpha .+ l_ij
        acc = acc .* Tacc.(alpha)

        v = ct.load(V, (1, j + 1i32, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)
    ct.store(O, (1, i, h, b), o → eltype(O))
    isnothing(M) || ct.store(M, (i, h, b), reshape(m_i, TILE_M))
    isnothing(L) || ct.store(L, (i, h, b), reshape(l_i, TILE_M))

    return
end

"""
    flex_attention!(O, Q, K, V; score_mod = NoOpScore(), mask_mod = FullMask(), kwargs...)

FlexAttention forward: fused multi-head attention whose variant is given by
two mods — `score_mod` rewrites the attention scores and `mask_mod` decides
which query-key pairs attend.

  * `Q`: `(Dk, SeqLen_Q, Heads, Batch)`
  * `K`: `(Dk, SeqLen_K, Heads_KV, Batch)`
  * `V`: `(Dv, SeqLen_K, Heads_KV, Batch)`
  * `O`: `(Dv, SeqLen_Q, Heads, Batch)`

`Heads` must be a multiple of `Heads_KV` (GQA).

Keywords:
  * `M`, `L`: optional `(SeqLen_Q, Heads, Batch)` Float32 buffers receiving
    the softmax row max and sum, required by [`∇flex_attention!`](@ref).
  * `block_mask`: a precomputed [`BlockMask`](@ref) for data-dependent sparsity.
  * `block_sparse = true`: in-kernel block skipping for masks with analytic
    geometry.
  * `input_pos = 0`: absolute position of the first query (KV-cache decoding).
  * `qk_scale = 1/√Dk`.
  * `TILE_M = 64`, `TILE_N = 64`: query/key tile sizes.
"""
function flex_attention!(O,
    Q, K, V;
    M = nothing,
    L = nothing,
    score_mod = NoOpScore(),
    mask_mod = FullMask(),
    block_mask::Optional{BlockMask} = nothing,
    block_sparse::Bool = true,
    input_pos::Integer = 0,
    qk_scale = nothing,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    TILE_M = 64,
    TILE_N = 64,
)
    Dq, SeqLen_Q, Heads, Batch = size(Q)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(K)
    Dv, SeqLen_V, Heads_V, Batch_V = size(V)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert Batch == Batch_K == Batch_V
    @assert size(O) == (Dv, SeqLen_Q, Heads, Batch)
    @assert iszero(Heads % Heads_KV)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(something(qk_scale, 1 / sqrt(Dk)))
    even_k = iszero(SeqLen_K % TILE_N)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    grid = (cld(SeqLen_Q, TILE_M), Heads * Batch)

    if block_mask isa BlockMask
        @cutile(blocks=grid,
            flex_fwd_bm(
                Q, K, V, O, M, L,
                block_mask.count, block_mask.idx, block_mask.full,
                score_mod, mask_mod,
                qk_scale, Int32(input_pos), Heads,
                tensorcore, accumulate,
                Constant(Dk_pow2),
                Constant(Dv_pow2),
                Constant(TILE_M),
                Constant(TILE_N),
                Constant(query_group_size),
                Constant(even_k),
            )
        )
    else
        eff_bs = block_sparse && analytic_useful(mask_mod)
        @cutile(blocks=grid,
            flex_fwd(
                Q, K, V, O, M, L,
                score_mod, mask_mod,
                qk_scale, Int32(input_pos), Heads,
                tensorcore, accumulate,
                Constant(Dk_pow2),
                Constant(Dv_pow2),
                Constant(TILE_M),
                Constant(TILE_N),
                Constant(query_group_size),
                Constant(even_k),
                Constant(eff_bs),
            )
        )
    end

    return O
end

#==============================================================================
 Backward — mha_bwd's structure (grid = Heads·Batch, outer KV loop, inner Q
 loop, Q̄ read-modify-write, register-accumulated K̄/V̄) with the variant hooks:
 score_mod re-applied for P, its VJP (`∇score`, mods.jl) for s̄, mask_mod in
 place of CAUSAL. Analytic block sparsity reuses the forward lattice
 transposed: for fixed kv block j, query block i is skipped when
 j ∉ kv_block_range(i) — sound because the range is a superset of touched
 blocks, so skipped pairs are fully masked (S̄ ≡ 0).

 The precomputed-BlockMask path has no backward yet (its index list is
 per-query-block; the outer-KV loop would need the transpose).
==============================================================================#

function flex_bwd(
    Q::TileArray4, K::TileArray4, V::TileArray4,
    Ō′::TileArray4,
    M::TileArray3{Float32},
    Δ::TileArray3{Float32},
    Q̄::TileArray4, K̄::TileArray4, V̄::TileArray4,
    score_mod, mask_mod, ∂score_mod,
    qk_scale::Float32, input_pos::Int32,
    H::Int,
    Tc::Type, Tacc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    EVEN_K::Bool, BLOCK_SPARSE::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    hb = ct.bid(1)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    q_len = size(Q, 2)
    k_len = size(K, 2)
    q_tiles = cld(q_len, Int32(TILE_M))
    kv_tiles = cld(k_len, Int32(TILE_N))
    TMi = Int32(TILE_M); TNi = Int32(TILE_N)

    for j in 0i32:kv_tiles-1i32
        k = ct.load(K, (1, j + 1i32, hₖ, b), (Dk, TILE_N); padding_mode)
        v = ct.load(V, (1, j + 1i32, hₖ, b), (Dv, TILE_N); padding_mode)

        k̄_acc = zeros(Tacc, (Dk, TILE_N))
        v̄_acc = zeros(Tacc, (Dv, TILE_N))

        kv_pos = BlockPos{1, TILE_N}(j)

        for i in 1i32:q_tiles
            iq = i - 1i32
            if BLOCK_SPARSE
                lo, hi = kv_block_range(mask_mod, iq, TMi, TNi, q_len, k_len, input_pos)
                visit = (j >= lo) & (j < hi)
            else
                visit = true
            end
            if visit
                q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode, allow_tma=false)
                ō = ct.load(Ō′, (1, i, h, b), (Dv, TILE_M); padding_mode, allow_tma=false)
                m = reshape(ct.load(M, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))
                δ = reshape(ct.load(Δ, (i, h, b), (TILE_M,), latency=1), (1, TILE_M))

                s₀ = muladd(transpose(k) → Tc, q → Tc, zeros(Tacc, (TILE_N, TILE_M)))
                s₀ = s₀ * Tacc(qk_scale)          # pre-mod scores — VJPs see these

                q_pos = BlockPos{2, TILE_M}(iq, input_pos)
                s = score_mod(s₀, b, h, q_pos, kv_pos)

                if BLOCK_SPARSE && kv_block_full(mask_mod, iq, j, TMi, TNi, q_len, k_len, input_pos)
                    if !EVEN_K
                        s = ifelse.(kv_pos .< k_len, s, Tacc(-Inf32))
                    end
                else
                    umask = mask_mod(b, h, q_pos, kv_pos)
                    if !EVEN_K
                        umask = umask .& (kv_pos .< k_len)
                    end
                    s = ifelse.(umask, s, Tacc(-Inf32))
                end

                safe = m .> -Inf32                # fully-masked column ⇒ p = 0
                p = ifelse.(safe, exp.(s .- Tacc.(m)), 0f0)

                v̄_acc = muladd(ō → Tc, transpose(p) → Tc, v̄_acc)

                p̄ = muladd(transpose(v) → Tc, ō → Tc, zeros(Tacc, (TILE_N, TILE_M)))
                ds = p .* (p̄ .- Tacc.(δ))         # S̄ w.r.t. post-mod scores

                s̄ = ∇score(score_mod, ∂score_mod, s₀, b, h, q_pos, kv_pos, ds)
                s̄ = s̄ * Tacc(qk_scale)

                q̄ = ct.load(Q̄, (1, i, h, b), (Dk, TILE_M), allow_tma=false)
                q̄ = muladd(k → Tc, s̄ → Tc, q̄ → Tacc)
                ct.store(Q̄, (1, i, h, b), q̄ → eltype(Q̄))

                k̄_acc = muladd(q → Tc, transpose(s̄) → Tc, k̄_acc)
            end
        end

        store = isone(QUERY_GROUP_SIZE) ? ct.store : atomic_add_tile
        store(K̄, (1, j + 1i32, hₖ, b), k̄_acc → eltype(K̄))
        store(V̄, (1, j + 1i32, hₖ, b), v̄_acc → eltype(V̄))
    end

    return
end

"""
    ∇flex_attention!(Q̄, K̄, V̄, Ō, Q, K, V, O, M, L; kwargs...)

Backward of [`flex_attention!`](@ref). The forward must be run with `M`/`L`
output buffers (each `(SeqLen_Q, Heads, Batch)` Float32). `score_mod` and
`mask_mod` must match the forward call. Parameter gradients of the score mod
are accumulated into `∂score_mod`, a [`grad_shadow`](@ref) of `score_mod`
(pass `nothing` to skip them). The precomputed-BlockMask path has no backward.
"""
function ∇flex_attention!(
    Q̄, K̄, V̄, Ō,
    Q, K, V, O, M, L;
    score_mod = NoOpScore(),
    mask_mod = FullMask(),
    ∂score_mod = nothing,
    block_sparse::Bool = true,
    input_pos::Integer = 0,
    qk_scale = nothing,
    tensorcore = tensorcore_type(eltype(Q)),
    accumulate = accumulate_type(tensorcore),
    TILE_M = 64,
    TILE_N = 64,
)
    Dq, SeqLen_Q, Heads, Batch = size(Q)
    Dk, SeqLen_K, Heads_KV, Batch_K = size(K)
    Dv, SeqLen_V, Heads_V, Batch_V = size(V)
    @assert Dq == Dk
    @assert SeqLen_K == SeqLen_V
    @assert Heads_KV == Heads_V
    @assert Batch == Batch_K == Batch_V
    @assert size(O, 1) == Dv
    @assert size(Ō, 1) == Dv
    @assert size(M) == size(L) == (SeqLen_Q, Heads, Batch)
    @assert iszero(Heads % Heads_KV)

    query_group_size = Heads ÷ Heads_KV
    qk_scale = Float32(something(qk_scale, 1 / sqrt(Dk)))
    even_k = iszero(SeqLen_K % TILE_N)
    Dk_pow2 = nextpow(2, Dk)
    Dv_pow2 = nextpow(2, Dv)

    Ō′, Δ = similar(Ō), similar(M)
    @cutile(blocks=(cld(SeqLen_Q, 32), Heads * Batch),
        mha_bwd_preprocess(
            Ō, O, Ō′, L, Δ,
            Constant(Heads),
            Constant(Dv_pow2),
            Constant(32),
        )
    )

    eff_bs = block_sparse && analytic_useful(mask_mod)
    @cutile(blocks=Heads * Batch,
        flex_bwd(
            Q, K, V, Ō′, M, Δ,
            fill!.((Q̄, K̄, V̄), 0)...,
            score_mod, mask_mod, ∂score_mod,
            qk_scale, Int32(input_pos), Heads,
            tensorcore, accumulate,
            Constant(Dk_pow2),
            Constant(Dv_pow2),
            Constant(TILE_M),
            Constant(TILE_N),
            Constant(query_group_size),
            Constant(even_k),
            Constant(eff_bs),
        )
    )

    return
end
