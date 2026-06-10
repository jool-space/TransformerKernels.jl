export softmax!
export ∇softmax!

function softmax_fwd(
    X::TileMatrix, Y::TileMatrix,
    TILE_M::Int
)
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))

    m = fill(-Inf32, TILE_M)
    s = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.NegInf) → Float32
        m_new = max.(m, x)
        safe = m_new .> -Inf32
        s = s .* ifelse.(safe, exp.(m .- m_new), 0f0) .+ ifelse.(safe, exp.(x .- m_new), 0f0)
        m = m_new
    end

    m_global = maximum(m)
    s_global = sum(s .* exp.(m .- m_global))

    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.NegInf) → Float32
        y = exp.(x .- m_global) ./ s_global
        ct.store(Y, (i, bid_n), y → eltype(Y))
    end

    return
end

function softmax_bwd(
    X̄::TileMatrix, Ȳ::TileMatrix,
    Y::TileMatrix,
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(Y, 1, (TILE_M, 1))

    acc = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        y = ct.load(Y, (i, bid_n), (TILE_M,); padding_mode) → Float32
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode) → Float32
        acc = y .* ȳ .+ acc
    end
    dot = sum(acc)

    for i in 1i32:num_tiles
        y = ct.load(Y, (i, bid_n), (TILE_M,); padding_mode) → Float32
        ȳ = ct.load(Ȳ, (i, bid_n), (TILE_M,); padding_mode) → Float32
        x̄ = y .* (ȳ .- dot)
        ct.store(X̄, (i, bid_n), x̄ → eltype(X̄))
    end

    return
end

function softmax_fwd_single(
    X::TileMatrix, Y::TileMatrix,
    TILE_M::Int
)
    bid_n = ct.bid(1)
    x = ct.load(X, (1i32, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.NegInf) → Float32
    p = exp.(x .- maximum(x))
    y = p ./ sum(p)
    ct.store(Y, (1i32, bid_n), y → eltype(Y))
    return
end

function softmax_bwd_single(
    X̄::TileMatrix, Ȳ::TileMatrix,
    Y::TileMatrix,
    TILE_M::Int
)
    bid_n = ct.bid(1)
    y = ct.load(Y, (1i32, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.Zero) → Float32
    ȳ = ct.load(Ȳ, (1i32, bid_n), (TILE_M,); padding_mode=ct.PaddingMode.Zero) → Float32
    x̄ = y .* (ȳ .- sum(y .* ȳ))
    ct.store(X̄, (1i32, bid_n), x̄ → eltype(X̄))
    return
end

const SOFTMAX_SINGLE_TILE_MAX = 4096
const SOFTMAX_STREAM_TILE = 1024

"""
    softmax!(Y, X)

Numerically stable softmax over each column of `X`. Columns up to 4096
elements run in a single tile; longer columns stream in two passes (online
max/sum, then normalize).
"""
function softmax!(Y::AbstractMatrix, X::AbstractMatrix)
    M, N = size(X)
    if M <= SOFTMAX_SINGLE_TILE_MAX
        @cutile(blocks=N, softmax_fwd_single(X, Y, Constant(nextpow(2, M))))
    else
        @cutile(blocks=N, softmax_fwd(X, Y, Constant(SOFTMAX_STREAM_TILE)))
    end
    return
end

"""
    ∇softmax!(X̄, Ȳ, Y)

Backward of [`softmax!`](@ref), from the forward OUTPUT `Y`:
`X̄ = Y .* (Ȳ .- sum(Y .* Ȳ))` per column.
"""
function ∇softmax!(X̄::AbstractMatrix, Ȳ::AbstractMatrix, Y::AbstractMatrix)
    M, N = size(Y)
    if M <= SOFTMAX_SINGLE_TILE_MAX
        @cutile(blocks=N, softmax_bwd_single(X̄, Ȳ, Y, Constant(nextpow(2, M))))
    else
        @cutile(blocks=N, softmax_bwd(X̄, Ȳ, Y, Constant(SOFTMAX_STREAM_TILE)))
    end
    return
end
