import typing

import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class QuantizedLatent(base.Latent):
    _num_values_per_latent: list[int]
    _values_per_latent: list[jnp.ndarray]
    optimize_values: bool
    bound_type: str
    _basis: jnp.ndarray
    codebook_size: int
    _codebook: jnp.ndarray

    def __init__(self, num_latents, num_values_per_latent, optimize_values, bound_type, key):
        values_key, _ = jax.random.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents
        self.optimize_values = optimize_values
        if not (bound_type == 'currrent' or bound_type == "normal"):
            self.bound_type = None
        else:
            self.bound_type = bound_type

        if isinstance(num_values_per_latent, int):
            self._num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self._num_values_per_latent = num_values_per_latent

        self._values_per_latent = []
        for i in range(num_latents):
            if self._num_values_per_latent[i] % 2 == 1:
                self._values_per_latent.append(jnp.linspace(-0.5, 0.5, self.num_values_per_latent[i]))
            else:
                self._values_per_latent.append(jnp.arange(self._num_values_per_latent[i])/self._num_values_per_latent[i] - 0.5)

        self._num_values_per_latent = self._num_values_per_latent
        self._basis = jnp.cumprod(jnp.array([1] + self._num_values_per_latent[:-1]))
        self.codebook_size = jnp.prod(self.num_values_per_latent).item()
        if self.codebook_size > 2**16 or self.codebook_size < 0:
            print("Warning: codebook size is larger than 2**16, which is not supported by the current implementation. Using lookup-free quantized latents instead.")
            self._codebook = None
        else:
            self._codebook = self.indices_to_codes(jnp.arange(self.codebook_size))

    @property
    def codebook(self):
        return jax.lax.stop_gradient(self._codebook)

    @property
    def basis(self):
        return jax.lax.stop_gradient(self._basis)
        
    @property
    def num_values_per_latent(self):
        #won't work if numebrs in num_values_per_latent are different
        return jax.lax.stop_gradient(jnp.array(self._num_values_per_latent))

    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [jax.lax.stop_gradient(v) for v in self._values_per_latent]
    
    # @staticmethod
    def bound_to_current(z, values, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        #num_values_per_latent = jnp.array([len(v) for v in values])
        bound_limit = jnp.array([[jnp.abs(v).max()] for v in values])
        
        return jnp.tanh(z) * bound_limit 
    
    @staticmethod
    def quantize(x, values):
        def distance(x, l):
            return jnp.abs(x - l)
        distances = jax.vmap(distance, in_axes=(None, 0))(x, values)
        index = jnp.argmin(distances)
        return values[index], index

    @staticmethod
    def bound(z):
        """Bound `z` to range -0.5 to 0.5 , an array of shape (..., d)."""
        return jnp.tanh(z) / 2

    def __call__(self, x, *, key=None):
        if self.bound_type:
            x = self.bound_to_current(x, self.values_per_latent) if self.bound_type == 'current' else self.bound(x)

        quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        quantized = jnp.stack([qi[0] for qi in quantized_and_indices])
        indices = jnp.stack([qi[1] for qi in quantized_and_indices])
        quantized_sg = x + jax.lax.stop_gradient(quantized - x)
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

    def indices_to_codes(
        self,
        indices
    ):
        """Inverse of `codes_to_indices`."""

        codes_non_centered = (indices[:, jnp.newaxis] // self.basis) % self.num_values_per_latent
        codes = self._scale_and_shift_inverse(codes_non_centered, self.num_values_per_latent)
        return codes


    @staticmethod
    def _scale_and_shift_inverse(zhat, values):
        half_width =values // 2
        return (zhat - half_width) / half_width / 2

    def sample(self, *, key):
        ret = []
        for values, subkey in zip(self.values_per_latent, jax.random.split(key, self.num_latents)):
            ret.append(jax.random.choice(subkey, values))
        return jnp.array(ret)
