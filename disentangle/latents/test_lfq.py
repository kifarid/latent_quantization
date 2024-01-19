import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
from .lookup_free_quantized import LFQuantizedLatent

def test_LFQuantizedLatent():
    # Test initialization
    num_latents = 3
    num_values_per_latent = [3, 4, 5]
    optimize_values = False
    key = jax.random.PRNGKey(0)
    min_max_range = [-1, 1]

    fs_quantized = LFQuantizedLatent(3, key=key)

    assert fs_quantized.num_latents == num_latents

    # Test sample method
    key = jax.random.PRNGKey(1)
    samples = fs_quantized.sample(key=key)
    assert isinstance(samples, jnp.ndarray)
    assert samples.shape == (num_latents,)

    #test all but with batch size
    batch_size = 3
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (batch_size, num_latents)) - 0.6
    outs = vmap(fs_quantized)(x)
    assert outs['z_continuous'].shape == (batch_size, num_latents)
    assert outs['z_quantized'].shape == (batch_size, num_latents)
    assert outs['z_hat'].shape == (batch_size, num_latents)
    assert outs['z_indices'].shape == (batch_size, num_latents)

    

test_LFQuantizedLatent()