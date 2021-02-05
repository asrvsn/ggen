''' Basic tests ''' 

import numpy as np
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn
import pdb

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    return array.reshape(array.shape[1]//nrows, nrows, -1, ncols).swapaxes(1, 2)

def normalize(self, X):
	''' Normalize batched inputs ''' 
	X_mu, X_std = jnp.mean(X, axis=1), jnp.std(X, axis=1)
	X_bar = (X - X_mu) / X_std
	return X_bar, X_mu, X_std

def unnormalize(self, X_bar, X_mu, X_std):
	return X_bar*X_std + X_mu

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

class Acquisition(nn.Module):
	'''
	Surrogate cost function taking AdjacencyOp parameters to a real value. 
	Architecture:
	1. Dense layers
	2. Bayesian linear regressor (BLR) layer
		- Trained initially as a dense (linear regression) layer
		- Learned basis functions parametrize BLR
	3. Expected improvement (or perhaps UCB)
		- use MCMC EI to calculated expected improvement
	'''
	alpha_init: float=1.0
	beta_init: float=1000

	def setup(self):
		self.alpha = self.param('alpha', lambda key, shape: self.alpha_init, ())
		self.beta = self.param('beta', lambda key, shape: self.beta_init, ())
		self.fc1 = nn.Dense(features=128)
		self.fc2 = nn.Dense(features=32)
		self.basis = lambda x: self.fc2(jnp.tanh(self.fc1(x)))
		self.final = nn.Dense(features=1)

	@nn.compact
	def __call__(self, x):
		''' Returns the predicted objective ''' 
		x = self.basis(x)
		x = self.final(x)
		return x

	def mll(self, X_bar, Y_bar):
		''' Marginal log-likelihood given normalized training data ''' 
		Phi = self.basis(X_bar)
		K = self.beta*Phi.T@Phi + self.alpha*jnp.eye(Phi.shape[1])
		M = self.beta*jnp.linalg.inv(K)@Phi.T@Y_bar
		n, d = self.X_bar.shape
		mll = (d/2.)*jnp.log(alpha) + \
				(n/2.)*jnp.log(beta) - \
				(n/2.) * jnp.log(2*jnp.pi) - \
        		(beta/2.) * jnp.linalg.norm(Y_bar - Phi@M, 2) - \
        		(alpha/2.) * M@M - \
        		(1/2.) * jnp.slogdet(K)[1]
        return mll, K, M

	def blr_predict(self, X_bar, K, M):
		''' Predicted mean and variance given normalized testing data'''
		Phi = self.basis(X_bar)
		mu = M.T@Phi
		var = Phi.T@jnp.linalg.inv(K)@Phi
		return mu, var

	def EI(self, X_bar, K, M, y_best):
		''' Expected-improvement acquisition value ''' 
		mu, var = self.blr_predict(X_bar, K, M)
		std = jnp.sqrt(var)
		gamma = (y_best - mu) / std
		ei = std * (gamma * jsp.stats.norm.cdf(gamma) + jsp.stats.norm.pdf(gamma))
		return ei

	def UCB(self, X_bar, K, M):
		''' Upper confidence bound acquisition value '''
		mu, var = self.blr_predict(X_bar, K, M)
		return mu + 0.01*jnp.sqrt(var)


def train(acq, X, Y, lr):
	''' 
	Train the acquisition function.
	'''
	X_bar, X_mu, X_std = normalize(X)
	Y_bar, Y_mu, Y_std = normalize(Y)
	# TODO

def maximize(acq, theta):
	'''
	Maximize trained acquisition function.
	Incorporates an additional cost component for L1 norm of scale parameter (preferring smaller graphs).
	'''
	pass

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
	key, subkey = jax.random.split(key)
	# phi = Acquisition().init(subkey)['params']

	mat, field = AdjacencyOp().apply({'params': theta})
	fig, axs = plt.subplots(1, 2)
	im = axs[0].imshow(mat)
	plt.colorbar(im, ax=axs[0])
	im = axs[1].imshow(field)
	plt.colorbar(im, ax=axs[1])
	plt.show()


