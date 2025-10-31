
from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mabrax import mabrax_env
from typing import Tuple

class MABraxAnt(Env):
    def __init__(self):
        self.env = mabrax_env.Ant()

    def get_obs(self, state, obs, target, xvel):
        qpos = state.pipeline_state.q[:2]
        obs_arr = jp.stack([obs[a] for a in self.env.agents])
#        jax.debug.print("vel: {}", xvel/self.env.env.dt)
        vel = jp.array(xvel)*1.0/self.env.env.dt
        jax.debug.print("xvel: {}, vel)
        return jp.concatenate((obs_arr, jp.repeat(qpos[None,:], self.env.num_agents, axis=0), jp.repeat(vel[None,None], self.env.num_agents, axis=0), 
                          jp.ones((self.env.num_agents, 1)))*5.0, axis=1)
                
    @property
    def observation_size(self) -> int:
        return self.env.observation_spaces[self.env.agents[0]].shape[0] + 2 + 2 #env obs + global pos + goal

    @property
    def action_size(self) -> int:
        return self.env.action_spaces[self.env.agents[0]].shape[0]

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")

    def reset(self, rng: jax.Array) -> State:
        """ Resets the environment to an initial state. """
        key1, key2, key3 = jax.random.split(rng, 3)
        
        # Reset the internal MPE environment
        obs, state = self.env.reset(key1)
        
        _, target = self._random_target(key3)

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
        info = {"seed": 0, "mpe_key": key2, "target": target}

        state = State(state, self.get_obs(state, obs, target, 0.0), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """ Run one timestep of the environment's dynamics. """
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # generate action dict for adversaries and good agents
        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        
        # step internal environment
        xpos0 = state.pipeline_state.pipeline_state.q[0]
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)
#        jax.debug.print("real vel: {}", (mpe_state.pipeline_state.q[0] - xpos0)/self.env.env.dt)

        # process obs into our format
        obs = self.get_obs(mpe_state, obs, state.info["target"], mpe_state.pipeline_state.q[0] - xpos0)
        
        # set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed, "mpe_key": key}
        state.info.update(info)
        
        reward, _ = jp.zeros(2)
        return state.replace(
            pipeline_state=mpe_state, obs=obs, reward=reward, done=list(dones.values())[0].astype(jp.float32)
        )
        
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 10
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])
