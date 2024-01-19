import typing

import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base

from typing import Optional


class LFQuantizedLatent(base.Latent):
    num_latents: int
    optimize_values: bool
    _mask: jnp.ndarray
    _codebook: jnp.ndarray
    should_bound: bool
    codebook_size: int

    def __init__(self,
                    num_latents: int,
                    should_bound: bool,
                    key: jax.random.PRNGKey):
            """
            Initialize the LookupFreeQuantized object.

            Args:
                num_latents (int): The dimension of the input, and codebook.
                key (jax.random.PRNGKey): The random key used for initialization.
            """
            values_key, _ = jax.random.split(key, 2)
            self.is_continuous = False
            self.num_latents = num_latents
            self.num_inputs = num_latents
            self.optimize_values = False
            codebook_size = 2 ** num_latents
            self.codebook_size = codebook_size
            self._mask = 2 ** jnp.arange(num_latents-1, -1, -1)
            all_codes = jnp.arange(codebook_size)
            bits = ((all_codes[..., None] & self._mask) != 0).astype(jnp.float32)
            self._codebook = self.bits_to_codes(bits)
            self.should_bound = should_bound
    
    @property
    def mask(self):
        return jax.lax.stop_gradient(self._mask)
    
    @property
    def codebook(self):
        return jax.lax.stop_gradient(self._codebook)

    def bits_to_codes(self, bits):
        return 2 * bits - 1
    
    @staticmethod
    def bound(z):
        """Bound `z`, an array of shape (..., d)."""
        return jnp.tanh(z)

    @staticmethod
    def quantize(x, mask):
        quantized = jnp.where(x > 0, 1, -1)
        indicies = jnp.sum((x > 0) * mask, axis=-1)
        return quantized, indicies
    
    def __call__(self, x, *, key=None):
        # quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        # quantized = jnp.stack([qi[0] for qi in quantized_and_indices])
        if self.should_bound:
            x = self.bound(x)
            
        quantized, indices = self.quantize(x, self.mask)
        quantized_sg = x + jax.lax.stop_gradient(quantized - x)
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

    def sample(self, *, key):
        ret = []
        for subkey in jax.random.split(key, self.num_latents):
            values = jnp.array([-1, 1])
            ret.append(jax.random.choice(subkey, values))
        return jnp.array(ret)
