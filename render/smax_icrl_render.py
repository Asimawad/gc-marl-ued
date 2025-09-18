import functools
import json
import os
import pickle
import distrax

import jax
import math
import wandb
from brax.io import model
from brax.io import html
from pyinstrument import Profiler

import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from brax import envs
from etils import epath
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
import tyro
from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.viz.visualizer import SMAXVisualizer

#get our environment
from envs.mpe_tag_facmac import MPETagFacmac
from envs.smax import SmaxEnv

from scipy.special import softmax

env = SmaxEnv()

#get our actor
class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"

    LOG_STD_MAX = 5
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

actor = Actor(action_size=env.action_size)

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

_, actor_params, _ = tyro.cli(load_params)

state_seq = []
import time
seed = 1
key = jax.random.PRNGKey(seed)
key_r, key = jax.random.split(key, 2)
state = env.reset(key_r)
max_steps = 100

for n in range(max_steps):
    _, key_step, key_discrete = jax.random.split(state.info["mpe_key"],3)

    # Iterate random keys and sample actions
    action, _ = actor.apply(actor_params, state.obs)
    action_ = None
    
    if env.env.action_type == 'continuous':
        action = nn.tanh(action)
        action_ = action.at[:,:].set((action[:,:]+1)/2.0)
        action_ = action_.at[:,1].set(0.0)
        action_ = action_.at[:,0].set(1.0)
    else:
#        action = jnp.ones_like(action) * 1000
#        action = action.at[0,1].set(-1000)
        # action = action - ((1-env.get_avail_actions(state))*1e8)
#        pi = distrax.Categorical(logits = action)
#        action_ = pi.sample(seed = key_discrete)
        action -= (1-env.get_avail_actions(state))*1e10
        action_ = jnp.argmax(action, axis=-1)

    state_seq.append((key_step, state.pipeline_state, {a: action_[i] for i,a in enumerate(env.env.agents)}))
    print('action: ', softmax(action, axis=-1), action_, n)
    print('obs size: ', env.observation_size, state.obs.shape)
    print('observation: ', state.obs[:, -6:])
#    print('observation (processed): ', state.obs[:, :])

    # Step environment
#    state = env.step(state, jnp.zeros((env.env.num_adversaries, env.action_size)))
    state = env.step(state, action)


viz = SMAXVisualizer(env.env, state_seq)
viz.animate(view=True)

