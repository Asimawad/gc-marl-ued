from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mabrax import mabrax_env
from typing import Tuple
from jaxmarl.wrappers.baselines import LogWrapper

class MABraxAnt(Env):
    def __init__(self):
        self.env = mabrax_env.Ant()
        self.env = LogWrapper(self.env)

    def get_obs(self, state, obs, target):
        qpos = state.pipeline_state.q[:2]
        obs_arr = jp.stack([obs[a] for a in self.env.agents])
        return jp.concatenate((obs_arr, jp.repeat(qpos[None,:], self.env.num_agents, axis=0), 
                          jp.repeat(target[None,:], self.env.num_agents, axis=0)), axis=1)
                
    @property
    def observation_size(self) -> int:
        return self.env.observation_spaces[self.env.agents[0]].shape[0] + 4 #env obs + global pos + goal

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
            "success_easy": zero,
        }
        info = {"seed": 0, "mpe_key": key2, "target": target, 'returned_episode_lengths': jp.zeros(self.env.num_agents),
                'returned_episode': jp.zeros(self.env.num_agents).astype(bool), 'returned_episode_returns': jp.zeros(self.env.num_agents),
                'episode_returns': jp.zeros(self.env.num_agents)}

        state = State(state, self.get_obs(state.env_state, obs, target), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """ Run one timestep of the environment's dynamics. """
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # generate action dict for adversaries and good agents
        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}

        pipeline_state = state.pipeline_state.replace(returned_episode_returns = state.info["returned_episode_returns"],
                                      episode_returns = state.info["episode_returns"],
                                      returned_episode_lengths = state.info["returned_episode_lengths"])
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)
        # process obs into our format
        obs = self.get_obs(mpe_state.env_state, obs, state.info["target"])
        
        # set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]

        #update info
        info = {"seed": seed, "mpe_key": key}
        state.info.update(info)
        state.info.update({'returned_episode_lengths': infos['returned_episode_lengths'], 
                           'returned_episode': infos['returned_episode'], 
                           'returned_episode_returns': infos['returned_episode_returns']})
        state.info.update({"episode_returns": mpe_state.episode_returns})

        #update metrics

        dist = jp.linalg.norm(mpe_state.env_state.pipeline_state.q[:2] - state.info["target"])
        success = jp.array(dist < 0.5, dtype=float)
        success_easy = jp.array(dist < 2., dtype=float)
        state.metrics.update({"dist": dist, "success": success, "success_easy": success_easy})

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
