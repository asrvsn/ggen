''' Basic tests ''' 

import numpy as np
import scipy as sp
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
import pdb

class AdjacencyKernel(nn.Module):
	@nn.compact
	def __call__(self, x):
		x = nn.DenseGeneral(features=128, axis=(-1,), bias_init=initializers.normal())(x)
		x = jnp.tanh(x)
		x = nn.DenseGeneral(features=64, axis=(-1,))(x)
		x = jnp.tanh(x)
		x = nn.DenseGeneral(features=32, axis=(-1,))(x)
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
		self.agg = Aggregator()

	@nn.compact
	def __call__(self):
		ksteps = jnp.round(1 / self.scale) 
		nsteps = ksteps * self.subscale
		grid_x, grid_y = jnp.meshgrid(jnp.linspace(0, 1, nsteps), jnp.linspace(0, 1, nsteps))
		grid_xy = jnp.stack((grid_x, grid_y), axis=2)
		field = self.ker(grid_xy).squeeze(axis=2)
		blocks = split(field, self.subscale, self.subscale)
		mat = self.agg(blocks).squeeze(axis=2)
		return mat, field

# TODO: try multiple version of AdjacencyOp, perhaps one where parameter space is sparser...
# - PDE w/ inputs or boundary conditions
# - density map of -1, 1 signals
# - 

def evaluate(op, theta):
	'''
	Evaluate black-box function on graph given parameters.
	'''
	mat, field = AdjacencyOp().apply({'params': theta})
	# Convert to graph representation & evaluate J

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	key = jax.random.PRNGKey(15)
	theta = AdjacencyOp().init(key)['params']

	mat, field = AdjacencyOp().apply({'params': theta})
	fig, axs = plt.subplots(1, 2)
	im = axs[0].imshow(mat)
	plt.colorbar(im, ax=axs[0])
	im = axs[1].imshow(field)
	plt.colorbar(im, ax=axs[1])
	plt.show()


