"""
    DNN

Module for representations of neural networks.

### Submodules

- [`Architecture`](@ref) -- data structures for neural networks
- [`FileFormats`](@ref)  -- IO of file representations of neural networks
"""
module DNN

using Reexport

include("Architecture/Architecture.jl")

include("FileFormats/FileFormats.jl")

@reexport using .Architecture
@reexport using .FileFormats

end  # module
