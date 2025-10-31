from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mabrax import mabrax_env
from typing import Tuple
from jaxmarl.wrappers.baselines import LogWrapper
from envs.mabrax_ant import MABraxAnt

class MABraxWalker(MABraxAnt):
    def __init__(self):
        self.env = mabrax_env.Walker2d()
        self.env = LogWrapper(self.env)
