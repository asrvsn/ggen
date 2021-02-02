''' Basic tests ''' 

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import pdb

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2))

class AdjacencyKernel(nn.Module):
	@nn.compact
	def __call__(self, x):
		x = nn.DenseGeneral(features=64, axis=(-1,))(x)
		x = jnp.tanh(x)
		x = nn.DenseGeneral(features=128, axis=(-1,))(x)
		x = jnp.tanh(x)
		x = nn.DenseGeneral(features=64, axis=(-1,))(x)
		x = jnp.tanh(x)
		x = nn.DenseGeneral(features=1, axis=(-1,))(x)
		x = jnp.tanh(x)
		return x

class Aggregator(nn.Module):
	alpha: float = 0.1

	@nn.compact
	def __call__(self, x):
		x = nn.DenseGeneral(features=1, axis=(-2, -1))(x) # {-1, 0, 1}
		x = 1.5*jnp.tanh(x/self.alpha) + 0.5*jnp.tanh(-3*x/self.alpha)
		x = jnp.round(x)
		return x

class AdjacencyOp(nn.Module):
	subscale: int = 10

	def setup(self):
		self.scale = self.param('scale', lambda key, shape: jax.random.uniform(key), ())
		self.ker = AdjacencyKernel()
		# self.vker = jax.vmap(jax.vmap(self.ker, 0, 0), 0, 0)
		self.agg = Aggregator()
		# self.vagg = jax.vmap(jax.vmap(self.agg, 0, 0), 0, 0)

	@nn.compact
	def __call__(self):
		ksteps = jnp.round(1 / jnp.asarray(self.scale)) 
		nsteps = ksteps * self.subscale
		grid_x, grid_y = jnp.meshgrid(jnp.linspace(0, 1, nsteps), jnp.linspace(0, 1, nsteps))
		grid_xy = jnp.stack((grid_x, grid_y), axis=2)
		field = self.ker(grid_xy).squeeze(axis=2)
		blocks = split(field, self.subscale, self.subscale)
		mat = self.agg(blocks).squeeze(axis=2)
		return mat, field

# Next: try silly loss comparing adjacency op w/ zero-padding with some target?

class Acquisition(nn.Module):
	pass

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	key = jax.random.PRNGKey(0)
	params = AdjacencyOp().init(key)['params']
	mat, field = AdjacencyOp().apply({'params': params})
	fig, axs = plt.subplots(1, 2)
	im = axs[0].imshow(mat)
	plt.colorbar(im, ax=axs[0])
	im = axs[1].imshow(field)
	plt.colorbar(im, ax=axs[1])
	plt.show()


