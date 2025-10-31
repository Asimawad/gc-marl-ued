import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import functools
import json
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
from envs.mabrax_ant import MABraxAnt
from IPython.display import HTML


print(jax.devices())

env = MABraxAnt()

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

state_seq = []
import time
key = jax.random.PRNGKey(int(time.time()))
key_r, key = jax.random.split(key, 2)
state = env.reset(key_r)
state.info['target'] = jnp.array([20,0])
max_steps = 1000

for _ in range(max_steps):
    state_seq.append(state.pipeline_state.env_state.pipeline_state)
    # Iterate random keys and sample actions
    key, key_s = jax.random.split(key, 2)
    action, _ = actor.apply(actor_params, state.obs)
    print('action: ', action)
    print('obs: ', state.obs)
    state = env.step(state, action)

url = html.render(env.env.env.sys.replace(dt = env.env.env.dt), state_seq, height=1024)
exp_dir =  'videos'
exp_name = 'ma_ant_test_' + vid_name
with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
    file.write(url)

