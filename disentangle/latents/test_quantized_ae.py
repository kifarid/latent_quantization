import jax
import jax.numpy as jnp
from jax import vmap
from jax.nn import softmax

def entropy(prob):
    return (-prob * jnp.log(prob)).sum(-1)

def entropy_loss(continuous, codebook, inv_temperature=0.5):
        
        def distance(x, l):
            # the same as euclidean distance up to a constant
            return -2 * jnp.sum(x * l)
        
        distance_to_each_code = vmap(distance, in_axes=(None, 0))
        # create vmapped function over all other dimensions except the last one
        for _ in range(continuous.ndim - 1):
            distance_to_each_code = vmap(distance_to_each_code, in_axes=(0, None))

        distance = distance_to_each_code(continuous, codebook)
        prob = jax.nn.softmax(-distance * inv_temperature, axis = -1)

        per_sample_entropy = jnp.mean(entropy(prob))

        # distribution over all available tokens in the batch

        num_dims = prob.ndim

        if num_dims < 3:
            axes = tuple(range(num_dims-1))
        axes = tuple(range(num_dims-2))
        avg_prob = prob.mean(axes)
        codebook_entropy = entropy(avg_prob).mean()

        # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
        # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

        return per_sample_entropy - codebook_entropy

def test_entropy_loss():
    # Test case 1
    # continuous shape: (2, 3, 4)
    # codebook shape: (5, 4)
    inv_temperature = 0.5

    loss = entropy_loss(continuous, codebook, inv_temperature)

    print(loss)
    # Test case 2
    continuous = jnp.array([[[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7], [0.7, 0.8, 0.9, 0.10]], [[0.10, 0.11, 0.12, 0.13], [0.13, 0.14, 0.15, 0.16], [0.16, 0.17, 0.18, 0.19]]])
    codebook = jnp.array([[0.7, 0.8, 0.9, 0.10], [0.10, 0.11, 0.12, 0.13], [0.13, 0.14, 0.15, 0.16], [0.16, 0.17, 0.18, 0.19], [0.19, 0.20, 0.21, 0.22]])

    inv_temperature = 1.0

    loss = entropy_loss(continuous, codebook, inv_temperature)

    print(loss)
    # Test case 3
    continuous = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    codebook = jnp.array([[0.7, 0.8, 0.9], [0.10, 0.11, 0.12]])
 
    inv_temperature = 2.0

    loss = entropy_loss(continuous, codebook, inv_temperature)

    print(loss)
    print("All test cases passed!")

test_entropy_loss()