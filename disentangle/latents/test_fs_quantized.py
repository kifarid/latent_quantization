import jax
import jax.numpy as jnp
import numpy as np
from .fs_quantized import FSQuantizedLatent

def test_FSQuantizedLatent():
    # Test initialization
    num_latents = 3
    num_values_per_latent = [3, 4, 5]
    optimize_values = False
    key = jax.random.PRNGKey(0)
    min_max_range = [-1, 1]

    fs_quantized = FSQuantizedLatent(num_latents, num_values_per_latent, optimize_values, key, min_max_range)

    assert fs_quantized.num_latents == num_latents
    assert fs_quantized.num_inputs == num_latents
    assert (fs_quantized.num_values_per_latent._value == num_values_per_latent).all()
    assert fs_quantized.min_max_range == min_max_range
    assert fs_quantized.optimize_values == optimize_values

    # Test values_per_latent property
    values_per_latent = fs_quantized.values_per_latent
    assert len(values_per_latent) == num_latents
    for i in range(num_latents):
        assert isinstance(values_per_latent[i], jnp.ndarray)
        assert values_per_latent[i].shape == (num_values_per_latent[i],)
        assert values_per_latent[i][0] == min_max_range[0]
        if num_values_per_latent[i] % 2 == 1:
            assert values_per_latent[i][-1] == min_max_range[1]

    # Test quantize method
    x = np.array([2.213, -1.9, 1])
    quantized, indices = fs_quantized.quantize(x, fs_quantized.num_values_per_latent)
    assert isinstance(quantized, jnp.ndarray)
    assert isinstance(indices, jnp.ndarray)
    assert quantized.shape == x.shape
    assert indices.shape == x.shape
    #assert np.allclose(quantized, np.array([0.0, -0.5, 1.0]))
    assert np.allclose(indices, np.array([2, 0, 4]))

    # Test bound method
    z = np.array([2.213, -1.9, 1])
    bound_z = fs_quantized.bound(z, fs_quantized.num_values_per_latent)
    assert isinstance(bound_z, jnp.ndarray)
    assert bound_z.shape == z.shape

    # Test sample method
    key = jax.random.PRNGKey(1)
    samples = fs_quantized.sample(key=key)
    assert isinstance(samples, jnp.ndarray)
    assert samples.shape == (num_latents,)

    #test all but with batch size
    batch_size = 3
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (batch_size, num_latents))
    outs = fs_quantized(x)
    assert outs['z_continuous'].shape == (batch_size, num_latents)
    assert outs['z_quantized'].shape == (batch_size, num_latents)
    assert outs['z_hat'].shape == (batch_size, num_latents)
    assert outs['z_indices'].shape == (batch_size, num_latents)

    

test_FSQuantizedLatent()