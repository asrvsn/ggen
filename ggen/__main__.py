''' Basic tests ''' 

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class AdjacencyKernel(nn.Module):
	@nn.compact
	def __call__(self, x):
		x = nn.Dense(features=64)(x)
		x = nn.relu(x)
		x = nn.Dense(features=128)(x)
		x = nn.relu(x)
		x = nn.Dense(features=64)(x)
		x = nn.relu(x)
		x = nn.Dense(features=1)(x)
		x = nn.relu(x)
		return x

class Aggregator(nn.Module):
	alpha: float = 1.0

	@compact
	def __call__(self, x):
		x = nn.Dense(features=3)(x) # {-1, 0, 1}
		x = 1.5*jnp.tanh(x/alpha) + 0.5*jnp.tanh(-3*x/alpha)
		x = jnp.round(x)
		return x

class AdjacencyOp(nn.Module):
	init_scale: int = 1
	subscale: int = 10

	def setup(self):
		self.scale = self.param('scale', lambda: self.init_scale)
		ker = AdjacencyKernel()
		ker = jax.vmap(ker, 0, 0)
		ker = jax.vmap(ker, 0, 0)
		self.ker = ker
		agg = Aggregator()
		self.agg = agg

	@compact
	def __call__(self):
		nsteps = jnp.round(1 / jnp.asarray(self.scale)) * self.subscale
		grid_x, grid_y = jnp.meshgrid(jnp.linspace(0, 1, nsteps), jnp.linspace(0, 1, nsteps))
		grid_xy = jnp.stack((grid_x, grid_y), axis=2)
		field = self.ker(grid_xy)
		# TODO how to apply agg to field?

if __name__ == '__main__':
	rng = jax.random.PRNGKey(0)

