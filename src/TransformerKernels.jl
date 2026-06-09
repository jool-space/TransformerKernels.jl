module TransformerKernels

import cuTile
import cuTile as ct
using cuTile: Constant, TFloat32, BFloat16

using CUDACore: @cuda, i32

macro cutile(args...)
    esc(:(@cuda backend=$cuTile $(args...)))
end

include("utils.jl")
include("attention/decode.jl")
include("softmax/softmax.jl")
include("norm/rms_norm.jl")
include("norm/layer_norm.jl")

end
