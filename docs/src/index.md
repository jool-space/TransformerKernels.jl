```@meta
CurrentModule = TransformerKernels
```

# TransformerKernels

Documentation for [TransformerKernels](https://github.com/jool-space/TransformerKernels.jl).

```@index
```

## Attention

```@docs
attention!
∇attention!
decode_attention!
```

## Softmax

```@docs
softmax!
∇softmax!
```

## Normalization

```@docs
rms_norm!
∇rms_norm
layer_norm!
∇layer_norm
```



## FlexAttention

```@docs
flex_attention!
∇flex_attention!
```

### Mask mods

```@docs
FullMask
CausalMask
SlidingWindowMask
PrefixMask
DocumentMask
AndMask
OrMask
prefix_lm
```

### Score mods

```@docs
NoOpScore
SoftCapScore
AliBiScore
BiasScore
ComposeScore
```

### Pair features

```@docs
PairFeatureScore
pair_feature
∇pair_feature
```

### Score mod gradients

```@docs
grad_shadow
```

### Block sparsity

```@docs
BlockMask
build_block_mask
```

### Host evaluation

```@docs
hmask
hscore
```
