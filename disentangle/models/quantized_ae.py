import collections

import ipdb
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import typing

import disentangle
from . import ae

def entropy(prob):
    return (-prob * jnp.log(prob)).sum(-1)

class QuantizedAE(ae.AE):

    @staticmethod
    def quantization_loss(continuous, quantized):
        return jnp.mean(jnp.square(jax.lax.stop_gradient(continuous) - quantized))
    
    @staticmethod
    def entropy_loss(continuous, codebook, inv_temperature=1.):
            
            def distance(x, l):
                # the same as euclidean distance up to a constant
                return -2 * jnp.sum(x * l)
            
            distance_to_each_code = jax.vmap(distance, in_axes=(None, 0))
            # create vmapped function over all other dimensions except the last one
            for _ in range(continuous.ndim - 1):
                distance_to_each_code = jax.vmap(distance_to_each_code, in_axes=(0, None))

            distance = distance_to_each_code(continuous, codebook)
            prob = jax.nn.softmax(-distance * inv_temperature, axis = -1)

            per_sample_entropy = jnp.mean(entropy(prob))

            # distribution over all available tokens in the batch

            num_dims = prob.ndim
            if num_dims < 2:
                raise ValueError(f'Expected prob to have at least 2 dimensions, got {num_dims}')
            else:
                axes = tuple(range(num_dims-1))
            avg_prob = prob.mean(axes)
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch
            return per_sample_entropy - codebook_entropy
    
    @staticmethod
    def commitment_loss(continuous, quantized):
        return jnp.mean(jnp.square(continuous - jax.lax.stop_gradient(quantized)))

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *args, key=None, **kwargs):
        outs = jax.vmap(model)(data['x'])
        
        if self.lambdas['quantization'] > 0:
            quantization_loss = jax.vmap(self.quantization_loss)(outs['z_continuous'], outs['z_quantized'])
        else:
            quantization_loss = 0.0

        if self.lambdas['commitment'] > 0:
            commitment_loss = jax.vmap(self.commitment_loss)(outs['z_continuous'], outs['z_quantized'])
        else:
            commitment_loss = 0.0
        
        if 'entropy' in self.lambdas.keys():
            if self.lambdas['entropy'] > 0:
                entropy_loss = self.entropy_loss(outs['z_continuous'],
                                                 model.latent.codebook,
                                                   self.lambdas['inv_temperature'] if 'inv_temperature' in self.lambdas else 1.0)
            else:
                entropy_loss = 0.0

        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'],
                                                                                           data['x'])
        partition_norms = self.partition_norms()

        loss = self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss + \
               self.lambdas['quantization'] * quantization_loss + \
               self.lambdas['commitment'] * commitment_loss

        if 'entropy' in self.lambdas.keys():
            loss += self.lambdas['entropy'] * entropy_loss

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
        }
        metrics.update({
            f'params_{k_partition}/{k_norm}': v
            for k_norm, v_norm in partition_norms.items() for k_partition, v in v_norm.items()
        })
        #            'quantization_loss': quantization_loss,
        #    'commitment_loss': commitment_loss,
        if self.lambdas['commitment'] > 0:
            metrics['commitment_loss'] = commitment_loss
        if self.lambdas['quantization'] > 0:
            metrics['quantization_loss'] = quantization_loss

        if 'entropy' in self.lambdas.keys():
            if self.lambdas['entropy'] > 0:
                metrics['entropy_loss'] = entropy_loss
            
        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return jnp.mean(loss), aux
import collections

import ipdb
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import typing

import disentangle
from . import ae


class QuantizedAE(ae.AE):

    @staticmethod
    def quantization_loss(continuous, quantized):
        return jnp.mean(jnp.square(jax.lax.stop_gradient(continuous) - quantized))

    @staticmethod
    def commitment_loss(continuous, quantized):
        return jnp.mean(jnp.square(continuous - jax.lax.stop_gradient(quantized)))

    @eqx.filter_jit
    def batched_loss(self, model, data, step, *args, key=None, **kwargs):
        outs = jax.vmap(model)(data['x'])
        quantization_loss = jax.vmap(self.quantization_loss)(outs['z_continuous'], outs['z_quantized'])
        commitment_loss = jax.vmap(self.commitment_loss)(outs['z_continuous'], outs['z_quantized'])
        binary_cross_entropy_loss = jax.vmap(disentangle.losses.binary_cross_entropy_loss)(outs['x_hat_logits'],
                                                                                           data['x'])
        partition_norms = self.partition_norms()

        loss = self.lambdas['binary_cross_entropy'] * binary_cross_entropy_loss + \
               self.lambdas['quantization'] * quantization_loss + \
               self.lambdas['commitment'] * commitment_loss

        metrics = {
            'loss': loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
            'quantization_loss': quantization_loss,
            'commitment_loss': commitment_loss,
        }
        metrics.update({
            f'params_{k_partition}/{k_norm}': v
            for k_norm, v_norm in partition_norms.items() for k_partition, v in v_norm.items()
        })

        aux = {
            'metrics': metrics,
            'outs': outs,
        }

        return jnp.mean(loss), aux
