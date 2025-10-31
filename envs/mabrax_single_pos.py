from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mabrax import mabrax_env
from jaxmarl.wrappers.position_based_reward import PositionAnt, PositionWalker
from typing import Tuple
from jaxmarl.wrappers.baselines import LogWrapper
from envs.mabrax_ant import MABraxAnt
from envs.mabrax_walker import MABraxWalker

class MABraxAnt_singlepos(MABraxAnt):
    def __init__(self):
        self.env = PositionAnt()
        self.env = LogWrapper(self.env)

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return rng, self.env.goal_pos

class MABraxWalker_singlepos(MABraxWalker):
    def __init__(self):
        self.env = PositionWalker()
        self.env = LogWrapper(self.env)

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return rng, self.env.goal_pos
