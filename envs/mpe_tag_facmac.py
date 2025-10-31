from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mpe import simple_facmac

from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper

class MPETagFacmac(Env):
    def __init__(self):
        self.env = simple_facmac.SimpleFacmacMPE3a()
        self.env = LogWrapper(self.env)

    def get_obs(self, state, obs):
        def get_agent_min(agent_obs):
            other_start = 4+2*self.env.num_landmarks
            adv_pos = agent_obs[other_start:other_start+2*(self.env.num_adversaries-1)]
            adv_pos = jp.concatenate((jp.zeros(2), adv_pos)) # add our own pos (0 bc rel pos)
            adv_pos = jp.reshape(adv_pos, (-1,2))
            good_pos = agent_obs[other_start+2*(self.env.num_adversaries-1):
                                        other_start+2*(self.env.num_agents-1)]
            good_pos = jp.reshape(good_pos, (-1,2))
            dist_mat = (adv_pos[:,None,:] - good_pos[None,:,:])**2
            dist_mat = jp.sum(dist_mat, axis=2)
            abs_good_pos = state.p_pos[self.env.num_adversaries:self.env.num_agents]
            good_hidden = jp.sum((abs_good_pos - good_pos - agent_obs[2:4])**2, axis=1) > 0.01
            dist_mat = jp.where(good_hidden, 99999999, dist_mat) # replace hidden prey with high value
            return jp.min(dist_mat)
        
        adv_obs = jp.array([obs[a] for a in self.env.adversaries])
        adv_min = jax.vmap(get_agent_min, in_axes=0)(adv_obs)
        return jp.concatenate((adv_obs, adv_min[:,None], 
                    jp.zeros((self.env.num_adversaries,1))), axis=1)
                
    @property
    def observation_size(self) -> int:
        # pos, vel, landmarks, other_pos, other_vel, dist, goal
        return 4 + 2*self.env.num_landmarks + 2*(self.env.num_agents-1) + 2*(self.env.num_good_agents) + 2

    @property
    def action_size(self) -> int:
        return 5 # from simple mpe documentation

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")

    def reset(self, rng: jax.Array) -> State:
        """ Resets the environment to an initial state. """
        key1, key2 = jax.random.split(rng, 2)
        
        # Reset the internal MPE environment
        obs, state = self.env.reset(key1)

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
        info = {"seed": 0, "mpe_key": key2, "mpe_target": 0, 'returned_episode_lengths': jp.zeros(self.env.num_agents),
                'returned_episode': jp.zeros(self.env.num_agents).astype(bool), 'returned_episode_returns': jp.zeros(self.env.num_agents),
                'episode_returns': jp.zeros(self.env.num_agents)}


        state = State(state, self.get_obs(state.env_state, obs), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """ Run one timestep of the environment's dynamics. """
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # generate action dict for adversaries and good agents
        actions = {agent: action[i] for i, agent in enumerate(self.env.adversaries)}
        actions.update({a: jp.zeros(self.action_size) for a in self.env.good_agents})

        # use the info to update the pipeline state
        # we need to do this because the brax env wrapper resets the pipeline state
        # whenever the episode ends, however log wrapper needs some of this information
        # to be persistent across episodes.
        pipeline_state = state.pipeline_state.replace(returned_episode_returns = state.info["returned_episode_returns"],
                                      episode_returns = state.info["episode_returns"],
                                      returned_episode_lengths = state.info["returned_episode_lengths"])
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, pipeline_state, actions)

        # process obs into our format
        obs = self.get_obs(mpe_state.env_state, obs)
        
        # set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed, "mpe_key": key}
        state.info.update(info)
        state.info.update(infos)
        state.info.update({"episode_returns": mpe_state.episode_returns})

        reward = jp.mean(jp.stack(list(rewards.values())[:self.env.num_adversaries]))
        return state.replace(
            pipeline_state=mpe_state, obs=obs, reward=reward, done=list(dones.values())[0].astype(jp.float32)
        )
