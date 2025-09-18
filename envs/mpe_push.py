from typing import Tuple
from brax import base
from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mpe import simple
from jaxmarl.environments.mpe.default_params import *

# TODO
# Change target to make adv. make good away from target
# essentially good target = target but adv_target = BIG_NUMBER - target

NUM_GOOD = 1
NUM_ADV = 1

class PushMPEMARL(Env):
    def __init__(self):
        self.env = simple.SimplePushMPE(num_good_agents=NUM_GOOD, 
                                        num_adversaries=NUM_ADV,
                                        action_type=CONTINUOUS_ACT)
        self.num_agents = NUM_GOOD + NUM_ADV

    def get_obs(self, state, obs, targets):
        def create_obs_good(ob, target):
            # Get the goal landmark's absolute position
            goal_pos = state.p_pos[self.env.num_agents + state.goal]
            
            # Convert relative position to absolute position for current state
            abs_pos = goal_pos - ob[2:4]  # takes goal_rel_position from observation
            
            # Convert landmark positions to absolute
            landmark_pos = state.p_pos[self.env.num_agents:self.env.num_agents + 
                                       self.env.num_landmarks].reshape(-1) - ob[5:9].reshape(-1)
            
            # Convert other agent position to absolute
            other_pos = state.p_pos[self.env.num_adversaries:self.env.num_agents].reshape(-1) - ob[11:13]
            
            # return jp.concatenate((ob[:2], abs_pos, target), axis=0)
            return jp.concatenate((
            ob[:2],         # self_vel (2)
            abs_pos,        # absolute position relative to goal (2)
            ob[4],          # goal_landmark_id (1)
            landmark_pos,   # absolute positions of all landmarks (4)
            ob[9:11],       # landmark_ids (2)
            other_pos,      # absolute position of other agent(s) (2 * num_agents)
            target          # target (TBD)
            ), axis=0)
        
        def create_obs_adv(ob, target):
            # Convert landmark positions to absolute
            landmark_pos = state.p_pos[self.env.num_agents:self.env.num_agents + self.env.num_landmarks].reshape(-1) - \
                      ob[2:6].reshape(-1)
        
            # Convert other agent position to absolute
            other_pos = state.p_pos[self.env.num_adversaries:self.env.num_agents].reshape(-1) -  ob[6:8]            
        
            # padding vars
            dummy_abs_pos, dummy_landmark_ids = jp.zeros(2), jp.zeros(2)

            return jp.concatenate((
            ob[:2],             # self_vel (2)
            dummy_abs_pos,      # zeros (2)
            0,                  # dummy id (1)
            landmark_pos,       # absolute positions of all landmarks (4)
            dummy_landmark_ids, # zeros (2)
            other_pos,          # absolute position of other agent(s) (2 * num_agents)
            target              # target (TBD)
            ), axis=0)

        # Process observations separately for good agents and adversaries
        processed_obs = {}
    
        # Process good agents
        for agent in self.env.good_agents:
            if agent in obs:
                processed_obs[agent] = create_obs_good(obs[agent], targets[agent])
        
        # Process adversaries
        for agent in self.env.adversaries:
            if agent in obs:
                processed_obs[agent] = create_obs_adv(obs[agent], targets[agent])
                
        # Convert dict to jax array
        return jp.stack(list(processed_obs.values()))

    @property
    def observation_size(self) -> int:
        # TODO what is size of target
        TBD = 30
        return 11 + (2 * self.num_agents) + TBD

    @property
    def action_size(self) -> int:
        return 5 # from Petting Zoo documentation

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        key1, key2, key3 = jax.random.split(rng, 3)
        
        # set the targets
        targets = self._random_target(key1)

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
        info = {"seed": 0, "mpe_key": key3, "mpe_target": targets}

        state = State(state, self.get_obs(state, obs, targets), reward, done, metrics)
        state.info.update(info)
        return state


    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        #generate action dict
        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)

        # process obs into our format
        obs = self.get_obs(mpe_state, obs, state.info["mpe_target"])
        
        #set trajectory id to differentiate between episodes
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
        """Returns a landmark location in a random position in the xy plane"""
        keys = jax.random.split(rng, self.env.num_agents)
        dist = 1
        ang = jp.pi * 2.0 * jax.vmap(jax.random.uniform)(keys)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        target_arr = jp.concatenate((target_x[:,None], target_y[:,None]), axis=1)
        return {a: target_arr[i] for i, a in enumerate(self.env.agents)}
