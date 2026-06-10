export layer_norm!
export ∇layer_norm

function layer_norm_fwd(
    X::TileMatrix{Float32}, W::TileVector{Float32},
    B::TileVector{Float32}, Y::TileMatrix{Float32},
    Mean::Optional{TileVector{Float32}}, Rstd::Optional{TileVector{Float32}},
    eps::Float32, TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    mean = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode)
        mean = mean .+ x
    end
    mean = sum(mean) / M
    isnothing(Mean) || (Mean[bid_n] = mean)

    var = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode)
        mask = ((i - 1i32) * Int32(TILE_M) .+ ct.arange(TILE_M)) .<= M
        centered_x = ifelse.(mask, x .- mean, 0.0f0)
        var = var .+ (centered_x .* centered_x)
    end
    var = sum(var) / M
    rstd = 1 / √(var + eps)
    isnothing(Rstd) || (Rstd[bid_n] = rstd)

    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,))
        w = ct.load(W, i, (TILE_M,))
        b = ct.load(B, i, (TILE_M,))
        y = (x .- mean) .* rstd
        y = y .* w .+ b
        ct.store(Y, (i, bid_n), y)
    end

    return
end

@inline function bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
    padding_mode = ct.PaddingMode.Zero
    x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode)
    w = ct.load(W, i, (TILE_M,); padding_mode)
    ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode)
    xhat = (x .- mean) .* rstd
    wȳ = w .* ȳ

    # Mask for valid elements
    indices = ct.arange(TILE_M)
    offset = (i - 1i32) * Int32(TILE_M)
    global_indices = offset .+ indices
    mask = global_indices .<= M

    xhat_masked = ifelse.(mask, xhat, 0.0f0)
    wȳ_masked = ifelse.(mask, wȳ, 0.0f0)

    return ȳ, xhat_masked, wȳ_masked
end

function layer_norm_bwd_dx(
    X̄::TileMatrix{Float32}, Ȳ::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    rstd = Rstd[bid_n]
    mean = Mean[bid_n]

    c1 = zeros(Float32, TILE_M)
    c2 = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        _, xhat, wȳ = bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
        c1 = c1 .+ (xhat .* wȳ)
        c2 = c2 .+ wȳ
    end
    c1 = sum(c1) / M
    c2 = sum(c2) / M

    for i in 1i32:num_tiles
        _, xhat, wȳ = bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
        x̄ = (wȳ .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(X̄, (i, bid_n), x̄)
    end

    return
end

function layer_norm_bwd_dx_partial_dwdb(
    X̄::TileMatrix{Float32}, Ȳ::TileMatrix{Float32},
    W̄::TileMatrix{Float32}, B̄::TileMatrix{Float32},
    X::TileMatrix{Float32}, W::TileVector{Float32},
    Mean::TileVector{Float32}, Rstd::TileVector{Float32},
    Locks::TileVector{Int},
    N_GROUPS::Int, TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)
    group_id = mod1(bid_n, Int32(N_GROUPS))

    mean = Mean[bid_n]
    rstd = Rstd[bid_n]

    c1 = zeros(Float32, TILE_M)
    c2 = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        _, xhat, wȳ = bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
        c1 = c1 .+ (xhat .* wȳ)
        c2 = c2 .+ wȳ
    end
    c1 = sum(c1) / M
    c2 = sum(c2) / M

    for i in 1i32:num_tiles
        ȳ, xhat, wȳ = bwd_helper(X, W, Ȳ, bid_n, i, mean, rstd, TILE_M, M)
        x̄ = (wȳ .- (xhat .* c1 .+ c2)) .* rstd
        ct.store(X̄, (i, bid_n), x̄)

        partial_w̄ = ȳ .* xhat
        partial_b̄ = ȳ

        # Acquire spinlock
        while ct.atomic_cas(Locks, group_id, 0, 1;
                memory_order=ct.MemoryOrder.Acquire) == 1
            # spin
        end

        # Critical section: accumulate partial gradients
        partial_w̄ = partial_w̄ .+ ct.load(W̄, (i, group_id), (TILE_M,))
        partial_b̄ = partial_b̄ .+ ct.load(B̄, (i, group_id), (TILE_M,))
        ct.store(W̄, (i, group_id), partial_w̄)
        ct.store(B̄, (i, group_id), partial_b̄)

        # Release spinlock
        ct.atomic_xchg(Locks, group_id, 0;
                      memory_order=ct.MemoryOrder.Release)
    end

    return
end

function layer_norm_bwd_dwdb(
    W̄::TileMatrix{Float32}, B̄::TileMatrix{Float32},
    FINAL_W̄::TileVector{Float32}, FINAL_B̄::TileVector{Float32},
    TILE_G::Int, TILE_F::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid = ct.bid(1)
    num_tiles = ct.num_tiles(W̄, 2, (TILE_F, TILE_G))

    w̄ = zeros(Float32, (TILE_F, TILE_G))
    b̄ = zeros(Float32, (TILE_F, TILE_G))
    for i in 1i32:num_tiles
        w̄ = w̄ .+ ct.load(W̄, (bid, i), (TILE_F, TILE_G); padding_mode)
        b̄ = b̄ .+ ct.load(B̄, (bid, i), (TILE_F, TILE_G); padding_mode)
    end
    ct.store(FINAL_W̄, bid, sum(w̄; dims=2))
    ct.store(FINAL_B̄, bid, sum(b̄; dims=2))

    return
end

"""
    layer_norm!(Y, X, W, B; eps, Mean = nothing, Rstd = nothing, TILE_M = 256)

Layer-normalize each column of `X`: `y = (x - mean) * rstd * w + b` with
`rstd = 1/√(var + eps)`.

  * `X`, `Y`: `(M, N)`
  * `W`, `B`: `(M,)`
  * `Mean`, `Rstd`: `(N,)`, optional statistics outputs, needed by [`∇layer_norm`](@ref)
"""
function layer_norm!(
    Y::AbstractMatrix,
    X::AbstractMatrix, W::AbstractVector, B::AbstractVector;
    Mean = nothing,
    Rstd = nothing,
    eps, TILE_M = 256,
)
    M, N = size(X)
    @assert length(W) == length(B) == M

    @cutile(blocks=N,
        layer_norm_fwd(X, W, B, Y, Mean, Rstd, Constant(Float32(eps)), Constant(TILE_M))
    )

    return
end

"""
    ∇layer_norm(Ȳ, X, W, B, Mean, Rstd; kwargs...) -> (X̄, W̄, B̄)

Backward of [`layer_norm!`](@ref); the forward must be run with `Mean`/`Rstd`
buffers.
"""
function ∇layer_norm(
    Ȳ::AbstractMatrix, X::AbstractMatrix,
    W::AbstractVector, B::AbstractVector,
    Mean::AbstractVector, Rstd::AbstractVector;
    N_GROUPS = 64,
    TILE_M = 256, TILE_F = 64, TILE_G = 64,
)
    M, N = size(X)

    X̄ = similar(X)
    W̄_partial = similar(W, M, N_GROUPS)
    B̄_partial = similar(B, M, N_GROUPS)
    Locks = similar(X, Int, N_GROUPS)
    W̄ = similar(W)
    B̄ = similar(B)

    @cutile(blocks=N,
        layer_norm_bwd_dx_partial_dwdb(
            X̄, Ȳ, fill!(W̄_partial, 0), fill!(B̄_partial, 0), X, W,
            Mean, Rstd, fill!(Locks, 0), Constant(N_GROUPS), Constant(TILE_M)
        )
    )

    @cutile(blocks=cld(M, TILE_F),
        layer_norm_bwd_dwdb(W̄_partial, B̄_partial, W̄, B̄, Constant(TILE_G), Constant(TILE_F))
    )

    return X̄, W̄, B̄
end
