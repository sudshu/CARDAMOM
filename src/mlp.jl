# DifferLand/optimization/forward.py + jax.nn.leaky_relu (negative_slope=0.01).
@inline leaky_relu(x) = ifelse(x >= 0f0, x, 0.01f0 * x)

@inline mlp_forward(layers::Tuple, x) = _mlp(layers, x)
@inline _mlp(l::Tuple{Any}, x) = l[1].Wt * x + l[1].b            # last layer: linear
@inline _mlp(l::Tuple, x) = _mlp(Base.tail(l), leaky_relu.(l[1].Wt * x + l[1].b))
