# always use the same instance of each activation function
const _id = Id()
const _relu = ReLU()
const _sigmoid = Sigmoid()
const _tanh = Tanh()

const available_activations = Dict(
    # Id
    "Id" => _id,
    "linear" => _id,
    "Linear" => _id,
    "Affine" => _id,
    # ReLU
    "relu" => _relu,
    "ReLU" => _relu,
    # Sigmoid
    "sigmoid" => _sigmoid,
    "Sigmoid" => _sigmoid,
    "σ" => _sigmoid,
    # Tanh
    "tanh" => _tanh,
    "Tanh" => _tanh
)
