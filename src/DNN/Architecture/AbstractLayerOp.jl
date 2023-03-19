abstract type AbstractLayerOp end

dim(L::AbstractLayerOp) = (dim_in(L), dim_out(L))
