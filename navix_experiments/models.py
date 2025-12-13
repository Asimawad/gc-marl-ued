import flax.linen as nn
import jax.numpy as jnp

from flax.linen.initializers import variance_scaling

class small_G_encoder(nn.Module):
    rep_size: int
    norm_type = "layer_norm"
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(g)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x

class sa_ConvEncoder(nn.Module):
    output_size:  int
    norm_type = "layer_norm"
    
    @nn.compact
    def __call__(self, x):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(16, kernel_size=(2, 2), kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(32, kernel_size=(2, 2), kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(64, kernel_size=(2, 2), kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.output_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)        
        return x