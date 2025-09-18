import jax
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jaxmarl.environments.mpe import MPEVisualizer
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax
from envs.mpe_tag_facmac import MPETagFacmac
import time

# Render parameters
PARAM_PATH = "actor_network_params_facmac_mod.pkl"
RANDOM_SEED = 8 # set to int(time.time()) for truly random
MAX_STEPS = 500

# Define IPPO network
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        # critic = activation(critic)
        # critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        # critic = activation(critic)
        # critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi

def load_params(path: str):
    '''
    Returns model parameters given a filepath string
    '''
    with open(path, 'rb') as f:
        params = pickle.load(f)
    return params

# Initialize environment and network
env = MPETagFacmac()
actor = ActorCritic(action_dim=env.action_size)
actor_params = load_params(PARAM_PATH)

state_seq = []
key = jax.random.PRNGKey(RANDOM_SEED)
key_r, key = jax.random.split(key, 2)
state = env.reset(key_r)
max_steps = MAX_STEPS

for _ in range(max_steps):
    # Save current state
    state_seq.append(state.pipeline_state)
    
    # Iterate random keys and sample actions
    key, key_s = jax.random.split(key, 2)
    action = actor.apply(actor_params, state.obs[:,:-2])
    action = action.sample(seed=key_s)
    print('action: ', action)
    print('obs: ', state.obs)

    # Step environment
    state = env.step(state, action)

viz = MPEVisualizer(env.env, state_seq)
viz.animate(view=True)
