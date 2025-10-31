from typing import Tuple
from brax import base
from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mpe import simple

class SimpleMPE(Env):
    def __init__(self):
        self.env = simple.SimpleMPE(action_type="Continuous")

    def get_obs(self, state, obs, target):
        obs = list(obs.values())[0]
        abs_pos = state.p_pos[1] - obs[2:] #subtract rel pos from landmark pos
        return jp.concatenate((obs[:2], abs_pos, target), axis=0)

    @property
    def observation_size(self) -> int:
        return 6 #vel, pos, target_pos

    @property
    def action_size(self) -> int:
        return 5 #from simple mpe documentation

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")
        return "There is no backend!!!"

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        key1, key2, key3 = jax.random.split(rng, 3)
        
        # set the target
        _, target = self._random_target(key1)

        # Reset the internal MPE environment
        obs, state = self.env.reset(key2)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        info = {"seed": 0, "mpe_key": key3, "mpe_target": target}

        state = State(state, self.get_obs(state, obs, target), reward, done, metrics)
        state.info.update(info)
        return state


    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        #generate action dict
        actions = {agent: action for i, agent in enumerate(self.env.agents)}
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)

        # convert obs into a jax array
        obs = self.get_obs(mpe_state, obs, state.info["mpe_target"])
        
        reward, _ = jp.zeros(2)
        info = {"mpe_key": key}
        state.info.update(info)
        return state.replace(
            pipeline_state=mpe_state, obs=obs, reward=reward, done=list(dones.values())[0].astype(jp.float32)
        )

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random position in the xy plane"""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 1
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])
