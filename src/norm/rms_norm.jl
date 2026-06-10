export rms_norm!
export ∇rms_norm

function rms_norm_fwd(
    X::TileMatrix, W::TileVector,
    Y::TileMatrix, Rstd::Optional{TileVector{Float32}},
    offset::Float32, eps::Float32,
    TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    ss = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.Zero) → Float32
        ss = ss .+ x .* x
    end
    rstd = 1 / √(sum(ss) / M .+ eps)
    isnothing(Rstd) || (Rstd[bid_n] = rstd)

    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,)) → Float32
        w = ct.load(W, i, (TILE_M,)) → Float32
        y = x .* rstd .* (w .+ offset)
        ct.store(Y, (i, bid_n), y → eltype(Y))
    end

    return
end

function rms_norm_bwd_dx_partial_dw(
    X̄::TileMatrix, Ȳ::TileMatrix,
    W̄::TileMatrix,
    X::TileMatrix, W::TileVector,
    Rstd::TileVector,
    Locks::TileVector{Int},
    offset::Float32, N_GROUPS::Int, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)
    group_id = mod1(bid_n, Int32(N_GROUPS))

    dd = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode)
        w = ct.load(W, i, (TILE_M,); padding_mode)
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode)
        dd = dd .+ (ȳ .* (w .+ offset) .* x)
    end
    dd = sum(dd) / M

    rstd = Rstd[bid_n]
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,))
        w = ct.load(W, i, (TILE_M,))
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,))

        x̄ = rstd .* (ȳ .* (w .+ offset) .- (rstd * rstd * dd) .* x)
        ct.store(X̄, (i, bid_n), x̄)

        partial_w̄ = ȳ .* x .* rstd

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_id, 0, 1;
                memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        partial_w̄ = partial_w̄ .+ ct.load(W̄, (i, group_id), (TILE_M,))
        ct.store(W̄, (i, group_id), partial_w̄)

        # Release spinlock
        ct.atomic_xchg(Locks, group_id, 0; memory_order=ct.MemoryOrder.Release)
    end

    return
end

function rms_norm_bwd_dw(
    W̄::TileMatrix{Float32},
    FINAL_W̄::TileVector{Float32},
    TILE_G::Int, TILE_F::Int
)
    bid = ct.bid(1)
    num_tiles = ct.num_tiles(W̄, 2, (TILE_F, TILE_G))

    w̄ = zeros(Float32, (TILE_F, TILE_G))
    for i in 1i32:num_tiles
        w̄ = w̄ .+ ct.load(W̄, (bid, i), (TILE_F, TILE_G); padding_mode=ct.PaddingMode.Zero)
    end
    ct.store(FINAL_W̄, bid, sum(w̄; dims=2))

    return
end

"""
    rms_norm!(Y, X, W; eps, offset = 0f0, Rstd = nothing, TILE_M = 256)

RMS-normalize each column of `X`: `y = x * rstd * (w + offset)` with
`rstd = 1/√(mean(x²) + eps)`.

  * `X`, `Y`: `(M, N)`
  * `W`: `(M,)`
  * `Rstd`: `(N,)`, optional `rstd` output, needed by [`∇rms_norm`](@ref)
"""
function rms_norm!(
    Y::AbstractMatrix, X::AbstractMatrix, W::AbstractVector;
    Rstd = nothing, eps, offset = 0.0f0, TILE_M = 256,
)
    M, N = size(X)

    @cutile(blocks=N, rms_norm_fwd(X, W, Y, Rstd, offset, eps, Constant(TILE_M)))

    return
end

"""
    ∇rms_norm(Ȳ, X, W, Rstd; offset = 0f0, kwargs...) -> (X̄, W̄)

Backward of [`rms_norm!`](@ref); the forward must be run with an `Rstd`
buffer, and `offset` must match.
"""
function ∇rms_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector, Rstd::AbstractVector;
    offset = 0f0, N_GROUPS = 64,
    TILE_M = 256, TILE_F = 64, TILE_G = 64,
)
    M, N = size(X)

    X̄ = similar(X)
    W̄_partial = similar(W, M, N_GROUPS)
    Locks = similar(X, Int, N_GROUPS)
    W̄ = similar(W)

    @cutile(blocks=N,
        rms_norm_bwd_dx_partial_dw(
            X̄, Ȳ, fill!(W̄_partial, 0), X, W, Rstd, fill!(Locks, 0),
            Constant(Float32(offset)), Constant(N_GROUPS), Constant(TILE_M)
        )
    )

    @cutile(blocks=cld(M, TILE_F),
        rms_norm_bwd_dw(W̄_partial, W̄, Constant(TILE_G), Constant(TILE_F))
    )

    return X̄, W̄
end
