import typing

import jax
import jax.numpy as jnp
import equinox as eqx

import disentangle
from . import base


class FSQuantizedLatent(base.Latent):
    _num_values_per_latent: list[int]
    _values_per_latent: list[jnp.ndarray]
    optimize_values: bool
    min_max_range: list[int]
    _basis: jnp.ndarray
    codebook_size: int
    _codebook: jnp.ndarray


    def __init__(self, num_latents, num_values_per_latent, optimize_values, key, min_max_range):
        values_key, _ = jax.random.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents

        if isinstance(num_values_per_latent, int):
            self._num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self._num_values_per_latent = num_values_per_latent

        self.min_max_range = min_max_range
        
        self._values_per_latent = []
        for i in range(num_latents):
            if self._num_values_per_latent[i] % 2 == 1:
                values_per_latent = jnp.linspace(self.min_max_range[0], self.min_max_range[1], self._num_values_per_latent[i])
            else:
                values_per_latent = jnp.arange(self._num_values_per_latent[i])/self._num_values_per_latent[i] * (self.min_max_range[1] - self.min_max_range[0]) + self.min_max_range[0]

            values_per_latent = values_per_latent if optimize_values else jax.lax.stop_gradient(values_per_latent)
            self._values_per_latent.append(values_per_latent)

        self.optimize_values = optimize_values
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
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [jax.lax.stop_gradient(v) for v in self._values_per_latent]
    
    @property
    def num_values_per_latent(self):
        #won't work if numebrs in num_values_per_latent are different
        return jax.lax.stop_gradient(jnp.array(self._num_values_per_latent))
                                     
    @staticmethod
    def quantize(x, values, bnd = 1.0):
        """Quantizes `x` to the nearest value in `values` by just rounding."""
        x_bnd = FSQuantizedLatent.bound(x, values) # from min to max 
        x_quant = jnp.round(x_bnd).astype(int)
        half_l = values // 2
        x_quant_normalized = x_quant / half_l  # from -1 to 1
        x_quant_normalized = x_quant_normalized * bnd

        index = FSQuantizedLatent.codes_to_indices_per_dim(x_quant_normalized, values)
        return x_quant, index


    def __call__(self, x, *, key=None):
        #scale x to -1 to 1 using tanh
        x = jnp.tanh(x) * 10
        quantized, indices = self.quantize(x, values = self.num_values_per_latent)
        quantized_sg = x + jax.lax.stop_gradient(quantized - x) #ste
        
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs
    
    @staticmethod
    def bound(z, values, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (values - 1) * (1 - eps) / 2
        offset = jnp.where(values % 2 == 0, 0.5, 0.0)
        shift = jnp.arctanh(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    @staticmethod
    def _scale_and_shift(zhat_normalized, values, bnd = 1.0):
        #zhat_normalized is from -bnd to bnd
        half_width = values // 2
        # from -bnd to bnd to -1 to 1
        zhat_normalized = zhat_normalized / bnd
        return (zhat_normalized * half_width) + half_width
    
    @staticmethod
    def _scale_and_shift_inverse(zhat, values, bnd = 1.0):
        half_width =values // 2

        #FROM -1 to 1
        zhat_normalized =  (zhat - half_width) / half_width
        # from -max to max
        return zhat_normalized * bnd
        #return (zhat - half_width) / half_width
    
    @staticmethod
    def codes_to_indices_per_dim(zhat, values):
        """Converts a `code` to an index in the codebook."""
        zhat = FSQuantizedLatent._scale_and_shift(zhat, values)
        return zhat.astype(int)

    def indices_to_codes(
        self,
        indices
    ):
        """Inverse of `codes_to_indices`."""

        codes_non_centered = (indices[:, jnp.newaxis] // self.basis) % self.num_values_per_latent
        codes = self._scale_and_shift_inverse(codes_non_centered, self.num_values_per_latent)
        return codes
    
    def sample(self, *, key):
        ret = []
        for values, subkey in zip(self.values_per_latent, jax.random.split(key, self.num_latents)):
            ret.append(jax.random.choice(subkey, values))
        return jnp.array(ret)
    


