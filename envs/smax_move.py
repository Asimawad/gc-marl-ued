from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.smax import heuristic_enemy_smax_env, map_name_to_scenario

class SmaxEnv(Env):
    def __init__(self):
        scenario = map_name_to_scenario("3m")
        self.env = heuristic_enemy_smax_env.HeuristicEnemySMAX(scenario=scenario, 
                            see_enemy_actions = True, walls_cause_death = True, attack_mode = "closest",
                            action_type = 'continuous')

    def get_obs(self, state, obs):
        ally_obs = jp.stack([obs[a] for a in self.env.agents])
        return jp.concatenate((ally_obs, state.state.unit_positions[:self.env.num_agents,:], jp.zeros((self.env.num_agents, 2))), axis=1)
                    
    def get_avail_actions(self, state: State):
        avail_act = self.env.get_avail_actions(state.pipeline_state)
        return jp.stack([avail_act[a] for a in self.env.agents])
                
    @property
    def observation_size(self) -> int:
        # environment observation plus enemy health and goal (0)
        return self.env.obs_size+4

    @property
    def action_size(self) -> int:
        if self.env.action_type == 'continuous':
             return self.env.action_spaces[self.env.agents[0]].shape[0]  #categorical actions for smax
        else:
             return self.env.action_spaces[self.env.agents[0]].n

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
        info = {"seed": 0, "mpe_key": key2, "mpe_target": 0}
        state = State(state, self.get_obs(state, obs), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """ Run one timestep of the environment's dynamics. """
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # generate action dict for agents
        if self.env.action_type == 'continuous':
            action = action.at[:,:3].set((action[:,:3]+1)/2.0)
            action = action.at[:,3].set((action[:,3]+1)*3.0)
        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)

        # process obs into our format
        obs = self.get_obs(mpe_state, obs)
        
        # set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed, "mpe_key": key}
        state.info.update(info)

        # check if the battle is won
        other_team_start_idx = self.env.num_allies
        won_battle = jp.all(
                jp.logical_not(
                    jax.lax.dynamic_slice_in_dim(
                        mpe_state.state.unit_alive, other_team_start_idx, self.env.num_enemies
                    )
                )
            )
        state.metrics.update({"success": jp.where(won_battle, 1.0, 0.0)})

        
        reward, _ = jp.zeros(2)
        return state.replace(
            pipeline_state=mpe_state, obs=obs, reward=reward, done=list(dones.values())[0].astype(jp.float32)
        )
