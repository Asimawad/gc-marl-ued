# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from dataclasses import replace as dataclass_replace
from functools import cached_property
from typing import Any, Dict, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.connector import Connector
from jumanji.environments.routing.connector.constants import (
    AGENT_INITIAL_VALUE,
    EMPTY,
    PATH,
    POSITION,
    TARGET,
)
from jumanji.environments.routing.connector.types import State as ConnectorState
from jumanji.types import TimeStep, StepType
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State


def aggregate_rewards(reward: chex.Array, num_agents: int) -> chex.Array:
    """Aggregate individual rewards across agents."""
    team_reward = jnp.sum(reward)
    return jnp.repeat(team_reward, num_agents)


@chex.dataclass
class ConnectorMarlState:
    """State wrapper for Connector that adds episode_seed tracking for ICRL.
    
    This is necessary for hindsight relabeling in ICRL buffer, which requires
    unique episode IDs to prevent mixing goals from different episodes.
    
    The `key` property forwards to the underlying ConnectorState.key so that
    AutoResetWrapper can access it for automatic resets.
    
    Additional fields for CRL tricks :
    - step_won: The step number when all agents connected (for win_repeat)
    - won_episode: Whether all agents have connected (cumulative flag)
    """
    state: ConnectorState
    episode_seed: int = 0
    step: int = 0
    step_won: float = 0.0
    won_episode: bool = False
    
    @property
    def key(self) -> chex.PRNGKey:
        """Forward key access to underlying state for AutoResetWrapper compatibility."""
        return self.state.key
    
    def replace(self, **kwargs) -> "ConnectorMarlState":
        """Create a new state with updated fields (JAX-compatible)."""
        return dataclass_replace(self, **kwargs)

@chex.dataclass
class VectorConnectorMarlState:
    """State wrapper for VectorConnector that adds episode_seed tracking for ICRL.
    
    This is necessary for hindsight relabeling in ICRL buffer, which requires
    unique episode IDs to prevent mixing goals from different episodes.
    
    The `key` property forwards to the underlying ConnectorState.key so that
    AutoResetWrapper can access it for automatic resets.
    
    Additional fields for CRL tricks :
    - step_won: The step number when all agents connected (for win_repeat)
    - won_episode: Whether all agents have connected (cumulative flag)
    """
    state: ConnectorState
    episode_seed: int = 0
    step: int = 0
    step_won: float = 0.0
    won_episode: bool = False
    
    @property
    def key(self) -> chex.PRNGKey:
        return self.state.key
    
    def replace(self, **kwargs) -> "VectorConnectorMarlState":
        return dataclass_replace(self, **kwargs)

class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool):
        self.add_global_state = add_global_state
        super().__init__(env)
        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        """The default way to create a global state for an environment if it has no
        available global state - concatenate all observations.
        """
        global_state = jnp.concatenate(obs.agents_view, axis=0)
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))
        return global_state

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep)
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return state, timestep.replace(observation=observation)

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        timestep = self.modify_timestep(timestep)
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return state, timestep.replace(observation=observation)

        return state, timestep

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )

        obs_spec = self._env.observation_spec
        obs_data = {
            "agents_view": obs_spec.agents_view,
            "action_mask": obs_spec.action_mask,
            "step_count": step_count,
        }

        if self.add_global_state:
            num_obs_features = obs_spec.agents_view.shape[-1]
            global_state = specs.Array(
                (self._env.num_agents, self._env.num_agents * num_obs_features),
                obs_spec.agents_view.dtype,
                "global_state",
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        return int(self._env.action_spec.num_values[0])

def switch_perspective(grid: chex.Array, agent_id: int, num_agents: int) -> chex.Array:
    """
    Encodes the observation with respect to the current agent defined by `agent_id`.
    Each agent sees its observations as values `1, 2, 3`. Observations of other agents
    are shifted cyclically based on their relative position. The mapping is designed
    such that the ordering of observations remains consistent.
    For example,in a 3-agent game, if we wanted to switch to agent 1's perspective, then:
    agent 1s values will change from 4,5,6 -> 1,2,3
    agent 2s values will change from 7,8,9 -> 4,5,6
    agent 0s values will change from 1,2,3 -> 7,8,9
    Agent 0 will be passed observations where it is represented by the values 1,2,3. Agent 1
    will be passed observations where it is represented by the values 1,2,3. However in the
    state agent 0 will always be 1,2,3 and agent 1 will always be 4,5,6."""
    new_grid = grid - AGENT_INITIAL_VALUE  # Center agent values around 0
    new_grid -= 3 * agent_id  # Move the obs
    new_grid %= 3 * num_agents  # Keep obs in bounds
    new_grid += AGENT_INITIAL_VALUE  # 'Un-center' agent obs around 0
    # Take agent values from rotated grid and empty values from old grid
    return jnp.where((grid >= AGENT_INITIAL_VALUE), new_grid, grid)

class ConnectorWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the MA Connector environment.

    Do not use the AgentID wrapper with this env, it has implicit agent IDs.
    
    This wrapper adds episode_seed tracking for ICRL hindsight relabeling.
    """

    def __init__(self, env: Connector, add_global_state: bool = False, aggregate_rewards: bool = True):
        super().__init__(env, add_global_state)
        self._env: Connector
        self._aggregate_rewards = aggregate_rewards
        self.agent_ids = jnp.arange(self.num_agents)

    def reset(self, key: chex.PRNGKey) -> Tuple[ConnectorMarlState, TimeStep]:
        """Reset the environment and initialize episode_seed and win tracking."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep, episode_seed=0, step=0, won_episode=False)
        
        # Wrap state with episode_seed and win tracking (matching SMAX implementation)
        marl_state = ConnectorMarlState(
            state=state, 
            episode_seed=0,
            step=0,
            step_won=jnp.array(0.0, dtype=jnp.float32),
            won_episode=False
        )
        
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return marl_state, timestep.replace(observation=observation)

        return marl_state, timestep


    def step(self, marl_state: ConnectorMarlState, action: chex.Array) -> Tuple[ConnectorMarlState, TimeStep]:
        """Step the environment with CRL tricks: win detection, state freezing, win_repeat."""
        # Step the base environment
        state, timestep = self._env.step(marl_state.state, action)
        
        # ==================== WIN DETECTION ====================
        won_this_step = (timestep.extras.get("ratio_connections", 0.0) == 1.0).astype(bool)
        new_won_episode_step = won_this_step
        state_won_episode = marl_state.won_episode | new_won_episode_step
        
        # ==================== STEP_WON TRACKING ====================
        step_won_current = jnp.array(marl_state.step_won, dtype=jnp.float32)
        zero_float = jnp.array(0.0, dtype=jnp.float32)
        
        step_won_after_reset = jax.lax.select(
            state_won_episode,
            step_won_current,
            zero_float
        )
        
        current_step_float = jnp.array(marl_state.step, dtype=jnp.float32)
        new_step_won = jax.lax.select(
            state_won_episode & (step_won_after_reset == zero_float),
            current_step_float,
            step_won_after_reset
        )
        
        # ==================== WIN REPEAT LOGIC ====================
        steps_since_won = current_step_float - new_step_won
        win_repeat = 5
        
        original_done = timestep.step_type == StepType.LAST
        new_done = jax.lax.select(state_won_episode, False, original_done)
        new_done = jax.lax.select(
            state_won_episode & (steps_since_won > win_repeat),
            True,
            new_done
        )
        
        time_limit_reached = marl_state.step >= self.time_limit
        new_done = jax.lax.select(time_limit_reached, True, new_done)
        new_step_type = jax.lax.select(new_done, StepType.LAST, StepType.MID)
        
        # ==================== STATE FREEZING ====================
        final_env_state = jax.tree_util.tree_map(
            lambda old, new: jax.lax.select(state_won_episode & ~won_this_step, old, new),
            marl_state.state,
            state
        )
        
        # ==================== REWARD MASKING ====================
        # Zero out rewards during frozen state to avoid contamination
        # Keep reward on the winning step, zero out for all subsequent frozen steps
        frozen_reward = jax.lax.select(
            state_won_episode & ~won_this_step,  # If frozen (won but not this exact step)
            jnp.zeros_like(timestep.reward),     # Zero reward
            timestep.reward                       # Keep original reward
        )
        
        # ==================== UPDATE TIMESTEP ====================
        timestep = self.modify_timestep(
            timestep.replace(step_type=new_step_type, reward=frozen_reward),
            episode_seed=marl_state.episode_seed,
            step=marl_state.step,
            won_episode=state_won_episode
        )
        
        # ==================== CREATE NEW STATE ====================
        new_marl_state = ConnectorMarlState(
            state=final_env_state,
            episode_seed=marl_state.episode_seed,
            step=marl_state.step + 1,
            step_won=new_step_won,
            won_episode=state_won_episode
        )

        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return new_marl_state, timestep.replace(observation=observation)

        return new_marl_state, timestep

    def modify_timestep(self, timestep: TimeStep, episode_seed: int = 0, step: int = 0, won_episode: bool = False) -> TimeStep[Observation]:
        """Modify the timestep for the Connector environment."""
        
        # TARGET = 3 = The number of different types of items on the grid.
        def create_agents_view(grid: chex.Array) -> chex.Array:
            grid = jax.vmap(switch_perspective, in_axes=(None, 0, None))(grid, self.agent_ids, self.num_agents)
            # Mark position and target of each agent with that agent's normalized index.
            positions = jnp.where(grid % TARGET == POSITION, jnp.ceil(grid / TARGET), 0) / self.num_agents
            targets = jnp.where((grid % TARGET == 0) & (grid != EMPTY), jnp.ceil(grid / TARGET), 0) / self.num_agents
            paths = jnp.where(grid % TARGET == PATH, 1, 0)
            position_per_agent = jnp.where(grid == POSITION, 1, 0)
            target_per_agent = jnp.where(grid == TARGET, 1, 0)
            agents_view = jnp.stack((positions, targets, paths, position_per_agent, target_per_agent), -1)
            # Flatten for ICRL: (num_agents, H, W, C) -> (num_agents, H*W*C)
            return agents_view.reshape(agents_view.shape[0], -1)

        obs_data = {
            "agents_view": create_agents_view(timestep.observation.grid),
            "action_mask": timestep.observation.action_mask,
            "step_count": jnp.repeat(step, self.num_agents),  # Use step from marl_state
        }

        # Check if episode is won (all agents connected)
        # Use won_battle from state (cumulative) instead of ratio_connections (momentary)
        won_episode = won_episode & (timestep.step_type == StepType.LAST)
        
        # Evaluator and ICRL expect per-agent arrays, not scalars!
        won_episode_per_agent = jnp.full((self.num_agents,), won_episode, dtype=bool)
        
        # Check if episode is truncated (time limit reached but not won)
        is_last = timestep.step_type == StepType.LAST
        # Use jnp.where for JAX tracing compatibility
        truncation = jnp.where(is_last & ~won_episode, 1.0, 0.0).astype(jnp.float32)

        # Add episode_seed and truncation to extras (required for ICRL)
        # Use astype instead of float() for JAX tracing compatibility
        metrics: Dict[str, Any] = {
            "seed": jnp.asarray(episode_seed, dtype=jnp.float32),
            "truncation": truncation,
            "env_metrics": {
                "won_episode": won_episode_per_agent,  # Per-agent array (num_agents,)
                **timestep.extras,
            }
        }

        # Whether or not aggregate the list of individual rewards.
        reward = timestep.reward
        if self._aggregate_rewards:
            reward = aggregate_rewards(reward, self.num_agents)
        return timestep.replace(observation=Observation(**obs_data), reward=reward, extras=metrics)

    def get_global_state(self, obs: Observation) -> chex.Array:
        """Constructs the global state from the global information
        in the agent observations (positions, targets and paths.)
        """
        # obs.agents_view is now flattened (num_agents, H*W*C)
        # Just concatenate all agent observations
        global_state = jnp.concatenate(obs.agents_view, axis=0)
        global_state = jnp.tile(global_state, (obs.agents_view.shape[0], 1))
        return global_state

    @cached_property
    def observation_spec(
        self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )
        # Flattened observation: (num_agents, grid_size * grid_size * 5)
        obs_dim = self._env.grid_size * self._env.grid_size * 5
        agents_view = specs.BoundedArray(
            shape=(self._env.num_agents, obs_dim),
            dtype=float,
            name="agents_view",
            minimum=0.0,
            maximum=1.0,
        )
        obs_data = {
            "agents_view": agents_view,
            "action_mask": self._env.observation_spec.action_mask,
            "step_count": step_count,
        }
        if self.add_global_state:
            # Global state is all agent observations concatenated and tiled
            global_obs_dim = obs_dim * self._env.num_agents
            global_state = specs.BoundedArray(
                shape=(self._env.num_agents, global_obs_dim),
                dtype=float,
                name="global_state",
                minimum=0.0,
                maximum=1.0,
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)

def _slice_around(pos: chex.Array, fov: int) -> Tuple[chex.Array, chex.Array]:
    """Return the start and length of a slice that when used to index a grid will
    return a 2*fov+1 x 2*fov+1 sub-grid centered around pos.

    Returns are meant to be used with a `jax.lax.dynamic_slice`
    """
    # Because we pad the grid by fov we need to shift the pos to the position
    # it will be in the padded grid.
    shifted_pos = pos + fov

    start_x = shifted_pos[0] - fov
    start_y = shifted_pos[1] - fov
    return start_x, start_y

# get location coordinates from 2D grid
def _get_location(grid: chex.Array) -> chex.Array:
    row_len = grid.shape[-1]
    index = jnp.argmax(grid)
    return jnp.asarray((jnp.floor(index / row_len), jnp.remainder(index, row_len)), dtype=int)

class VectorConnectorWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the Connector environment.

    This wrapper transforms the grid-based observation to a vector of features. This env should
    have the AgentID wrapper applied to it since there is not longer a channel that can encode
    AgentID information.
    """

    def __init__(self, env: Connector, add_global_state: bool = False, aggregate_rewards: bool = True):
        self.fov = 2
        super().__init__(env, add_global_state)
        self._env: Connector
        self._aggregate_rewards = aggregate_rewards
        self.agent_ids = jnp.arange(self.num_agents)

    def modify_timestep(self, timestep: TimeStep, episode_seed: int = 0, step: int = 0, won_episode: bool = False) -> TimeStep[Observation]:
        """Modify the timestep for the Connector environment."""

        # TARGET = 3 = The number of different types of items on the grid.
        def create_agents_view(grid: chex.Array) -> chex.Array:
            grid = jax.vmap(switch_perspective, in_axes=(None, 0, None))(grid, self.agent_ids, self.num_agents)
            positions = jnp.where(grid % TARGET == POSITION, True, False)
            targets = jnp.where((grid % TARGET == 0) & (grid != EMPTY), True, False)
            paths = jnp.where(grid % TARGET == PATH, True, False)

            # group positions and paths
            blockers = jnp.where(positions, 1, jnp.where(paths, -1, 0))

            position_per_agent = grid == POSITION
            target_per_agent = grid == TARGET

            # group agents own target and other targets
            combined_targets = jnp.where(target_per_agent, 1, jnp.where(targets, -1, 0))

            # get coordinates of each agent's location and target
            position_coords = jax.vmap(_get_location)(position_per_agent)
            target_coords = jax.vmap(_get_location)(target_per_agent)

            def _create_one_agent_view(i: int) -> chex.Array:
                slice_len = 2 * self.fov + 1, 2 * self.fov + 1
                slice_x, slice_y = _slice_around(position_coords[i], self.fov)
                padded_blockers = jnp.pad(blockers[i], self.fov, constant_values=True)

                blockers_around_agent = jax.lax.dynamic_slice(padded_blockers, (slice_x, slice_y), slice_len)
                blockers_around_agent = jnp.reshape(blockers_around_agent, -1).astype(float)

                my_pos = position_coords[i] / grid[0].size
                my_target = target_coords[i] / grid[0].size

                padded_combined_targets = jnp.pad(combined_targets[i], self.fov, constant_values=True)

                targets_around_agent = jax.lax.dynamic_slice(padded_combined_targets, (slice_x, slice_y), slice_len)
                targets_around_agent = jnp.reshape(targets_around_agent, -1).astype(float)

                return jnp.concatenate(
                    [my_pos, my_target, blockers_around_agent, targets_around_agent],
                    dtype=float,
                )

            return jax.vmap(_create_one_agent_view)(jnp.arange(self.num_agents))

        obs_data = {
            "agents_view": create_agents_view(timestep.observation.grid),
            "action_mask": timestep.observation.action_mask,
            "step_count": jnp.repeat(step, self.num_agents),
        }
        
        # Add ICRL-required extras
        won_episode = won_episode & (timestep.step_type == StepType.LAST)
        won_episode_per_agent = jnp.full((self.num_agents,), won_episode, dtype=bool)
        is_last = timestep.step_type == StepType.LAST
        truncation = jnp.where(is_last & ~won_episode, 1.0, 0.0).astype(jnp.float32)
        
        metrics: Dict[str, Any] = {
            "seed": jnp.asarray(episode_seed, dtype=jnp.float32),
            "truncation": truncation,
            "env_metrics": {
                "won_episode": won_episode_per_agent,
                **timestep.extras,
            }
        }

        reward = timestep.reward
        if self._aggregate_rewards:
            reward = aggregate_rewards(reward, self.num_agents)
        return timestep.replace(observation=Observation(**obs_data), reward=reward, extras=metrics)

    @cached_property
    def observation_spec(
        self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )
        # 2 sets of tiles in fov (blockers and targets) + xy position of agent and target
        tiles_in_fov = (self.fov * 2 + 1) ** 2
        single_agent_obs = 4 + tiles_in_fov * 2
        agents_view = specs.BoundedArray(
            shape=(self.num_agents, single_agent_obs),
            dtype=float,
            name="agents_view",
            minimum=-1.0,
            maximum=1.0,
        )
        obs_data = {
            "agents_view": agents_view,
            "action_mask": self._env.observation_spec.action_mask,
            "step_count": step_count,
        }
        if self.add_global_state:
            global_state = specs.BoundedArray(
                shape=(self.num_agents, self.num_agents * single_agent_obs),
                dtype=float,
                name="global_state",
                minimum=-1.0,
                maximum=1.0,
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)

    def reset(self, key: chex.PRNGKey) -> Tuple[VectorConnectorMarlState, TimeStep]:
        """Reset with episode_seed tracking."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep, episode_seed=0, step=0, won_episode=False)
        
        marl_state = VectorConnectorMarlState(
            state=state,
            episode_seed=0,
            step=0,
            step_won=jnp.array(0.0, dtype=jnp.float32),
            won_episode=False
        )
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return marl_state, timestep.replace(observation=observation)

        return marl_state, timestep
    def step(self, marl_state: VectorConnectorMarlState, action: chex.Array) -> Tuple[VectorConnectorMarlState, TimeStep]:
        """Step the environment with CRL tricks: win detection, state freezing, win_repeat."""
        # Step the base environment
        state, timestep = self._env.step(marl_state.state, action)
        
        # ==================== WIN DETECTION ====================
        # Option 1: Use ratio_connections from Jumanji's extras
        # Jumanji Connector computes: extras["ratio_connections"] = jnp.mean(state.agents.connected)
        won_this_step = (timestep.extras.get("ratio_connections", 0.0) == 1.0).astype(bool)
        
        # Option 2 (alternative): Use state.agents.connected directly
        # won_this_step = jnp.all(state.agents.connected).astype(bool)
        
        new_won_episode_step = won_this_step
        state_won_episode = marl_state.won_episode | new_won_episode_step
        
        # ==================== STEP_WON TRACKING ====================
        step_won_current = jnp.array(marl_state.step_won, dtype=jnp.float32)
        zero_float = jnp.array(0.0, dtype=jnp.float32)
        
        step_won_after_reset = jax.lax.select(
            state_won_episode,
            step_won_current,
            zero_float
        )
        
        current_step_float = jnp.array(marl_state.step, dtype=jnp.float32)
        new_step_won = jax.lax.select(
            state_won_episode & (step_won_after_reset == zero_float),
            current_step_float,
            step_won_after_reset
        )
        
        # ==================== WIN REPEAT LOGIC ====================
        steps_since_won = current_step_float - new_step_won
        win_repeat = 5
        
        original_done = timestep.step_type == StepType.LAST
        new_done = jax.lax.select(state_won_episode, False, original_done)
        new_done = jax.lax.select(
            state_won_episode & (steps_since_won > win_repeat),
            True,
            new_done
        )
        
        time_limit_reached = marl_state.step >= self.time_limit
        new_done = jax.lax.select(time_limit_reached, True, new_done)
        new_step_type = jax.lax.select(new_done, StepType.LAST, StepType.MID)
        
        # ==================== STATE FREEZING ====================
        final_env_state = jax.tree_util.tree_map(
            lambda old, new: jax.lax.select(state_won_episode & ~won_this_step, old, new),
            marl_state.state,
            state
        )
        
        # ==================== REWARD MASKING ====================
        # Zero out rewards during frozen state to avoid contamination
        # Keep reward on the winning step, zero out for all subsequent frozen steps
        frozen_reward = jax.lax.select(
            state_won_episode & ~won_this_step,  # If frozen (won but not this exact step)
            jnp.zeros_like(timestep.reward),     # Zero reward
            timestep.reward                       # Keep original reward
        )
        
        # ==================== UPDATE TIMESTEP ====================
        timestep = self.modify_timestep(
            timestep.replace(step_type=new_step_type, reward=frozen_reward),
            episode_seed=marl_state.episode_seed,
            step=marl_state.step,
            won_episode=state_won_episode
        )
        
        # ==================== CREATE NEW STATE ====================
        new_marl_state = VectorConnectorMarlState(
            state=final_env_state,
            episode_seed=marl_state.episode_seed,
            step=marl_state.step + 1,
            step_won=new_step_won,
            won_episode=state_won_episode
        )

        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return new_marl_state, timestep.replace(observation=observation)

        return new_marl_state, timestep