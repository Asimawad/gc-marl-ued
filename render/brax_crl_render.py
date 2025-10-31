import functools
import json
import os
import pickle

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

#get our environment
from envs.ant import Ant
env = Ant(
    backend="spring",
    exclude_current_positions_from_observation=False,
    terminate_when_unhealthy=True,
)

#get our actor
class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"

    LOG_STD_MAX = 2
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

def load_params(path: str, vid_name: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return (pickle.loads(buf), vid_name)

(_, actor_params, _), vid_name = tyro.cli(load_params)

inference_fn = actor.apply
jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)
jit_inference_fn = jax.jit(inference_fn)

rollout = []
rng = jax.random.PRNGKey(seed=2)
state = jit_env_reset(rng=rng)
for i in range(5000):
    rollout.append(state.pipeline_state)
    act, _ = jit_inference_fn(actor_params, state.obs) #needs to be our actor nn
    state = jit_env_step(state, act)
    if i % 1000 == 0:
        state = jit_env_reset(rng=rng)

url = html.render(env.sys.replace(dt=env.dt), rollout, height=1024)
exp_dir =  'videos'
exp_name = 'ant_test_' + vid_name
with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
    file.write(url)
