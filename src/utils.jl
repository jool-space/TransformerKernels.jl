using cuTile: TileArray

const TileVector{T} = TileArray{T,1}
const TileMatrix{T} = TileArray{T,2}
const TileArray3{T} = TileArray{T,3}
const TileArray4{T} = TileArray{T,4}
const TileArray5{T} = TileArray{T,5}

const Optional{T} = Union{T,Nothing}

x → T::Type = T.(x)

struct TransposePostfix end

const ᵀ = TransposePostfix()

Base.:(*)(x, ::TransposePostfix) = transpose(x)

arithmetic_type(T::Type) = T
arithmetic_type(::Type{TFloat32}) = Float32

tensorcore_type(T::Type) = T
tensorcore_type(::Type{Float32}) = TFloat32

accumulate_type(T::Type) = T
accumulate_type(::Type{TFloat32}) = Float32
accumulate_type(::Type{BFloat16}) = Float32
accumulate_type(::Type{Float16}) = Float16
