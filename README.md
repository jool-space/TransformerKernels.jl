# TransformerKernels

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/TransformerKernels.jl/dev/)
[![Build Status](https://github.com/jool-space/TransformerKernels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/TransformerKernels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/TransformerKernels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/TransformerKernels.jl)

Transformer building blocks as GPU kernels written in
[cuTile.jl](https://github.com/JuliaGPU/cuTile.jl), NVIDIA's tile-based
programming model for Julia: fused multi-head attention, split-KV decoding,
softmax, and layer/RMS normalization, with backward passes throughout.
