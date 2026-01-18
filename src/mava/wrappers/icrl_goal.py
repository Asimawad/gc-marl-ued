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
from typing import Tuple, Dict, Any
from functools import cached_property

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep
from jumanji.environments.routing.connector.types import State as ConnectorState

from dataclasses import replace as dataclass_replace
from jumanji.environments.routing.connector.types import State as ConnectorState
from jumanji.types import StepType
from mava.wrappers.jumanji import VectorConnectorWrapper
from mava.types import Observation

@chex.dataclass
class ICRLVectorConnectorState:
    """State wrapper for ICRL VectorConnector with CRL tricks.
    
    Combines:
    - state: Connector state
    - episode_seed: For hindsight relabeling in ICRL buffer

    """
    state: ConnectorState
    episode_seed: int = 0 # For hindsight relabeling in ICRL buffer
    
    @property
    def key(self) -> chex.PRNGKey:
        """Forward key access to underlying state for AutoResetWrapper compatibility."""
        return self.state.key
    
    def replace(self, **kwargs) -> "ICRLVectorConnectorState":
        """Create a new state with updated fields (JAX-compatible)."""
        return dataclass_replace(self, **kwargs)


class ICRLVectorConnectorWrapper(VectorConnectorWrapper):
    """ICRL wrapper for VectorConnector that inherits from VectorConnectorWrapper.
    
    """
    
    def __init__(
        self,
        env,
        add_global_state: bool = False,
        aggregate_rewards: bool = True,
        goal_type: str = "distance_per_agent",
        obs_dim: int = 0,
        goal_dim: int = 2,
        goal_start_idx: int = 0,
        goal_end_idx: int = 2,
    ):
        """Initialize ICRL VectorConnector wrapper.

        """
        super().__init__(env, add_global_state, aggregate_rewards)
        
        # ICRL-specific parameters
        self.goal_type = goal_type
        self.base_obs_dim = obs_dim
        self._goal_dim = goal_end_idx - goal_start_idx
        self.goal_start_idx = goal_start_idx
        self.goal_end_idx = goal_end_idx
        
        # Select goal extraction function
        if goal_type == "total_distance":
            self._extract_goal = self._goal_total_distance
        elif goal_type == "distance_per_agent":
            self._extract_goal = self._goal_distance_per_agent
        elif goal_type == "ratio_connected":
            self._extract_goal = self._goal_ratio_connected
        elif goal_type == "position_coords":
            self._extract_goal = self._goal_position_coords
        else:
            raise ValueError(f"Unknown goal_type: {goal_type}")
    
    def reset(self, key: chex.PRNGKey) -> Tuple[ICRLVectorConnectorState, TimeStep]:
        """Reset with episode_seed tracking."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep, episode_seed=0)
        
        icrl_state = ICRLVectorConnectorState(
            state=state,
            episode_seed=0,

        )
        
        # Append goal to observation
        timestep = self._append_goal_to_obs(icrl_state, timestep)
        
        if self.add_global_state:
            from mava.types import ObservationGlobalState
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return icrl_state, timestep.replace(observation=observation)

        return icrl_state, timestep
    
    def step(self, icrl_state: ICRLVectorConnectorState, action: chex.Array) -> Tuple[ICRLVectorConnectorState, TimeStep]:
        """Step the environment with ICRL tricks."""
        # Step the base environment
        state, timestep = self._env.step(icrl_state.state, action)
   
        timestep = self.modify_timestep(
            timestep,
            episode_seed=icrl_state.episode_seed,

        )
        
        new_icrl_state = ICRLVectorConnectorState(
            state=state,
            episode_seed=icrl_state.episode_seed,
        )

        # Append goal to observation
        timestep = self._append_goal_to_obs(new_icrl_state, timestep)

        if self.add_global_state:
            from mava.types import ObservationGlobalState
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return new_icrl_state, timestep.replace(observation=observation)

        return new_icrl_state, timestep
    
    def modify_timestep(self, timestep: TimeStep, episode_seed: int = 0): # step: int = 0, won_episode: bool = False) -> TimeStep[Observation]:
        """Modify the timestep for the Connector environment - adds ICRL extras."""
        
        
        # Call parent's modify_timestep to get base observations (without our ICRL parameters)
        # We need to temporarily override to pass it through
        timestep_base = super().modify_timestep(timestep)
        
        # Now update with ICRL-specific fields
        obs_data = {
            "agents_view": timestep_base.observation.agents_view,
            "action_mask": timestep_base.observation.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self.num_agents),  
        }
        
        # Add ICRL-required extras
        won_episode = timestep.extras.get("ratio_connections", 0.0) == 1.0
        is_last = timestep.step_type == StepType.LAST
        truncation = jnp.where(is_last & ~won_episode, 1.0, 0.0).astype(jnp.float32)
        
        new_extras = {
            "seed": jnp.asarray(episode_seed, dtype=jnp.float32),
            "truncation": truncation, # not used
        }
        metrics: Dict[str, Any] = {**timestep_base.extras, **new_extras}

        return timestep_base.replace(observation=Observation(**obs_data), extras=metrics)
    
    def _append_goal_to_obs(
        self,
        icrl_state: ICRLVectorConnectorState,
        timestep: TimeStep
    ) -> TimeStep:
        """Extract goal and append to observation."""
        # Extract goal using selected goal function
        goal = self._extract_goal(icrl_state)
        
        # Append goal to observation
        # agents_view shape: (num_agents, obs_dim)
        new_agents_view = jnp.concatenate(
            [timestep.observation.agents_view, goal],
            axis=-1
        )
        
        # Create new observation with goal appended (Observation is a NamedTuple, use _replace)
        new_observation = timestep.observation._replace(agents_view=new_agents_view)
        
        return timestep.replace(observation=new_observation)
    
    # =====================================================================
    # Goal Extraction Functions 
    # =====================================================================
    
    def _goal_total_distance(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Goal = [mean_distance, 1.0] where mean_distance is normalized mean Manhattan distance."""
        state = icrl_state.state
        
        positions = state.agents.position
        targets = state.agents.target
        
        distance = jnp.sum(jnp.abs(positions - targets), axis=-1)
        mean_distance = jnp.mean(distance)
        
        grid_size = state.grid.shape[0]
        normalized_distance = mean_distance / (2 * grid_size)
        
        current_progress = 1.0 - normalized_distance
        ultimate_goal = 1.0
        
        goal = jnp.stack([current_progress, ultimate_goal], axis=0)
        goal = jnp.tile(goal, (self.num_agents, 1))
        
        return goal
    
    def _goal_distance_per_agent(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Goal = [per_agent_distance, 1.0] where each agent gets its own distance."""
        state = icrl_state.state
        
        positions = state.agents.position
        targets = state.agents.target
        
        distance = jnp.sum(jnp.abs(positions - targets), axis=-1)
        
        grid_size = state.grid.shape[0]
        normalized_distances = distance / (2 * grid_size)
        
        current_progress = 1.0 - normalized_distances
        ultimate_goal = jnp.ones((self.num_agents,))
        
        goal = jnp.stack([current_progress, ultimate_goal], axis=-1)
        
        return goal
    
    def _goal_ratio_connected(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Goal = [ratio_connected, 1.0] where ratio_connected is fraction of agents connected."""
        state = icrl_state.state
        
        ratio_connected = jnp.mean(state.agents.connected.astype(jnp.float32))
        
        ultimate_goal = 1.0
        
        goal = jnp.stack([ratio_connected, ultimate_goal], axis=0)
        goal = jnp.tile(goal, (self.num_agents, 1))
        
        return goal
    
    def _goal_position_coords(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Goal = [x, y] normalized position coordinates (like navix).
        
        This is the simplest goal representation - just the (x, y) position of each agent.
        The goal encoder learns to embed these 2D coordinates.
        During hindsight relabeling, future positions become goals.
        """
        state = icrl_state.state
        
        # Get current position of each agent: [num_agents, 2]
        positions = state.agents.position.astype(jnp.float32)
        targets = state.agents.target.astype(jnp.float32)
        goal = jnp.concatenate([positions, targets], axis=-1)
        return goal  # [num_agents, 4]
    
    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Observation spec with goal dimension added."""
        base_spec = super().observation_spec
        
        old_agents_view_spec = base_spec.agents_view
        old_shape = old_agents_view_spec.shape
        
        # Add goal_dim to the last dimension
        new_shape = (int(old_shape[0]), int(old_shape[1]) + self._goal_dim)
        
        new_agents_view_spec = specs.BoundedArray(
            shape=new_shape,
            dtype=old_agents_view_spec.dtype,
            minimum=old_agents_view_spec.minimum,
            maximum=old_agents_view_spec.maximum,
            name=old_agents_view_spec.name,
        )
        
        # Replace agents_view spec
        return base_spec.replace(agents_view=new_agents_view_spec)