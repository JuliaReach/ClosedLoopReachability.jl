import ReachabilityAnalysis: split

struct Splitter{S, M, O}
    split_fun::S
    merge_fun::M
    output_type::O
end

split(s::Splitter, X) = s.split_fun(X)
merge(s::Splitter, Xs) = s.merge_fun(Xs)

function NoSplitter(output_type=LazySet{Float64})
    split_fun = X0 -> [X0]
    function merge_fun(Xs)
        length(array(Xs)) != 1 && error("unexpected input")
        return first(array(Xs))
    end
    return Splitter(split_fun, merge_fun, output_type)
end

function BoxSplitter(partition=nothing, output_type=LazySet{Float64})
    if partition == nothing
        split_fun = X -> split(box_approximation(X), 2 * ones(Int, dim(X)))
    else
        split_fun = X -> split(box_approximation(X), partition)
    end
    merge_fun = Xs -> box_approximation(Xs)
    return Splitter(split_fun, merge_fun, output_type)
end
