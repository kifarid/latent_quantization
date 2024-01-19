import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
from .quantized import QuantizedLatent
from disentangle.models import base


def test_QuantizedLatent():
    # Test initialization
    num_latents = 3
    num_values_per_latent = [3, 4, 5]
    optimize_values = False
    key = jax.random.PRNGKey(0)

    fs_quantized = QuantizedLatent(num_latents, num_values_per_latent, optimize_values, False, key)

    assert fs_quantized.num_latents == num_latents
    assert fs_quantized.num_inputs == num_latents
    assert (fs_quantized.num_values_per_latent._value == num_values_per_latent).all()
    assert fs_quantized.optimize_values == optimize_values

    # Test values_per_latent property
    values_per_latent = fs_quantized.values_per_latent
    assert len(values_per_latent) == num_latents
    for i in range(num_latents):
        assert isinstance(values_per_latent[i], jnp.ndarray)
        assert values_per_latent[i].shape == (num_values_per_latent[i],)
        assert values_per_latent[i][0] == -0.5

    # Test quantize method
    x = np.array([2.213, -1.9, 1])

    # Test sample method
    key = jax.random.PRNGKey(1)
    samples = fs_quantized.sample(key=key)
    assert isinstance(samples, jnp.ndarray)
    assert samples.shape == (num_latents,)

    #test all but with batch size
    batch_size = 3
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (batch_size, num_latents)) 
    outs = vmap(fs_quantized)(x)
    assert outs['z_continuous'].shape == (batch_size, num_latents)
    assert outs['z_quantized'].shape == (batch_size, num_latents)
    assert outs['z_hat'].shape == (batch_size, num_latents)
    assert outs['z_indices'].shape == (batch_size, num_latents)

    

test_QuantizedLatent()