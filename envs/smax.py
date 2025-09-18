from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.smax import heuristic_enemy_smax_env, map_name_to_scenario
from jaxmarl.environments.smax import idle_enemy_smax_env
import distrax

class SmaxEnv(Env):
    def __init__(self, map_name="2s3z"):
        scenario = map_name_to_scenario(map_name)
        #modified to idle
        self.env = heuristic_enemy_smax_env.HeuristicEnemySMAX(scenario=scenario, 
                            see_enemy_actions = True, walls_cause_death = True, attack_mode = "closest",
                            action_type = 'discrete') 

                            #unit_type_attack_ranges=jp.array([100.0]*6),
                            #unit_type_sight_ranges=jp.array([100.0]*6))

    def get_obs(self, state, obs):
        def get_agent_enemy_healths(agent_obs):
            num_feat = len(self.env.unit_features)
            enemy_features = agent_obs[(self.env.num_agents-1)*num_feat:(self.env.num_agents-1)*num_feat+self.env.num_enemies*num_feat]
            enemy_features = jp.reshape(enemy_features, (self.env.num_enemies,-1))
            enemy_healths = enemy_features[:,0]
            max_healths = self.env.unit_type_health[state.state.unit_types[jp.arange(self.env.num_enemies)+self.env.num_agents]]
            enemy_healths_real = state.state.unit_health[self.env.num_agents:self.env.num_agents+self.env.num_enemies]
            enemy_healths_real = enemy_healths_real / max_healths
            enemy_healths_obs = jp.where((enemy_healths-enemy_healths_real)**2<0.01, enemy_healths, 1.0)
            enemy_healths_obs = jp.where(jp.all(agent_obs == 0), enemy_healths_real, enemy_healths_obs) # because we want agents to be able to observe enemy healths after they die
            return jp.sum(enemy_healths_obs)
            # return enemy_healths_obs
        
        ally_obs = jp.stack([obs[a] for a in self.env.agents])
        # ally_enemy_healths = jax.vmap(get_agent_enemy_healths)(ally_obs)
        # return jp.concatenate((ally_obs, ally_enemy_healths, jp.zeros((self.env.num_agents,self.env.num_enemies))), axis=1)\
        
        ally_enemy_healths = jax.vmap(get_agent_enemy_healths)(ally_obs)[:,None]
        return jp.concatenate((ally_obs, ally_enemy_healths, jp.zeros((self.env.num_agents,1))), axis=1)

                    
    def get_avail_actions(self, state: State):
        avail_act = self.env.get_avail_actions(state.pipeline_state)
        return jp.stack([avail_act[a] for a in self.env.agents])
                
    @property
    def observation_size(self) -> int:
        # environment observation plus enemy health and goal (0)
        return self.env.obs_size+2
        # return self.env.obs_size+self.env.num_enemies*2

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
        info = {"seed": 0, "mpe_key": key2, "mpe_target": 0, "world_state": obs['world_state'], 'step_won': 0}
        state = State(state, self.get_obs(state, obs), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """ Run one timestep of the environment's dynamics. """
        key, key_s, key_discrete = jax.random.split(state.info["mpe_key"], 3)
        
        # generate action dict for agents
        if self.env.action_type == 'continuous':
            action = action.at[:,:].set((action[:,:]+1)/2.0)
            # action = action.at[:,:3].set((action[:,:3]+1)/2.0)
            # action = action.at[:,3].set((action[:,3]+1)*3.0)
            action = action.at[:,1].set(0.0)  # Set do_shoot = 1.0 AFTER the scaling
            action = action.at[:,0].set(1.0)  # Set shoot_last_enemy to 0.0 for first agent
        else:
            # pi = distrax.Categorical(logits = action)
            # action = pi.sample(seed = key_discrete)
            action = jp.argmax(action, axis=-1)

        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        
        # step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)

        # process obs into our format
        world_state = obs['world_state']
        obs = self.get_obs(mpe_state, obs)
        
        # edge case: health doesn't immediately reset at done
        #obs = jp.where(rewards[self.env.agents[0]] > 0, obs.at[:,-2].set(0.0), obs)
        
        # set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed, "mpe_key": key, "world_state": world_state}
        state.info.update(info)

        # check if the battle is won
        won_battle = jp.where(rewards[self.env.agents[0]] > 0, 1, 0)
        state.info.update({'step_won': jp.where(won_battle, state.info['step_won'], 0.0)})
        step_won = jp.where(jp.logical_and(won_battle, jp.logical_not(state.info['step_won'])), 
                        state.info['steps'], state.info['step_won'])
        state.info.update({'step_won': step_won})
        state.metrics.update({"success": jp.where(won_battle, 1.0, 0.0)})
        
        steps_since_won = state.info['steps'] - state.info['step_won']
        win_repeat = 5
        
        reward, _ = jp.zeros(2)
        done = jp.where(won_battle, 0.0, dones['__all__'])
        done = jp.where(jp.logical_and(won_battle, steps_since_won > win_repeat), 1.0, done)
        obs = jp.where(won_battle, state.obs.at[:,-2].set(0.0), obs)
        pipeline_state = jax.tree_map(
            lambda x, y: jax.lax.select(won_battle, x, y), state.pipeline_state, mpe_state
        )
        return_state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return return_state
        
        # return state.replace(
        #     pipeline_state=mpe_state, obs=obs, reward=reward, done=dones['__all__'].astype(jp.float32)
        # )
