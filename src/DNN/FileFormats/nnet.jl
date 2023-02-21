# ===========
# NNET format
# ===========

"""
    read_nnet(file; final_activation=Id())

Read a neural network stored in a `.nnet` file.

### Input

- `file`             -- string indicating the location of the `.nnet` file
- `final_activation` -- (optional, default: `Id()`) activation function of the
                        last layer

### Output

A `FeedforwardNetwork` struct.

### Notes

For more info on the `.nnet` format, see [here](https://github.com/sisl/NNet).
The format assumes all hidden layers have ReLU activation except for the last
one.

The code is taken from
[here](https://github.com/sisl/NeuralVerification.jl/blob/237f0924eaf1988188219aff4360a677534f3871/src/utils/util.jl#L10).
"""
function read_nnet(file::String; final_activation=Id())
    f = open(file)
    line = readline(f)
    while occursin("//", line)  # skip comments
        line = readline(f)
    end
    nlayers = parse(Int64, split(line, ",")[1])
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    for i in 1:5
        line = readline(f)
    end
    layers = Layer[_read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, _read_layer(last(layer_sizes), f, final_activation))
    return FeedforwardNetwork(layers)
end

function _read_layer(output_dim::Int64, f::IOStream, act = ReLU())
     rowparse(splitrow) = parse.(Float64, splitrow[findall(!isempty, splitrow)])
     W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
     weights = vcat(W_str_vec'...)
     bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
     bias = rowparse(bias_string)
     return Layer(weights, bias, act)
end
