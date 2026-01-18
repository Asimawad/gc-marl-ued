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

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Tuple, Union, Dict, Any
from functools import cached_property

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
from jaxmarl.environments.mpe.simple import State as MPEState
from jaxmarl.environments.smax.heuristic_enemy_smax_env import State as SMAXState
from jumanji.environments.routing.connector.types import State as ConnectorState
from jumanji.environments.routing.connector.constants import AGENT_INITIAL_VALUE, POSITION, TARGET

from mava.types import MarlEnv, Observation, ObservationGlobalState, State
from mava.wrappers.jaxmarl import JaxMarlState


class ICRLGoalWrapper(Wrapper):
    """Wrapper that appends goal information to observations for ICRL.
    
    This wrapper replicates the exact goal extraction logic from the original ICRL implementation:
    - MPE Tag: Appends minimum distance to nearest visible prey for each predator
    - SMAX: Appends sum of visible enemy healths for each agent
    
    The goal is computed from both state (ground truth) and observation (partial observability).
    """
    def __init__(self, env: MarlEnv, env_name: str, obs_dim: int, goal_dim: int, goal_start_idx: int, goal_end_idx: int, goal_type: str = "manhattan_distance"):
        """Initialize the ICRL goal wrapper.
        
        Args:
            env: The environment to wrap (should be MPEWrapper or SmaxWrapper)
            env_name: Name of the environment ("Smax", "Connector")
            obs_dim: Base observation dimension
            goal_dim: Goal dimension (always 2)
            goal_start_idx: Index where goal starts in observation
            goal_end_idx: Index where goal ends in observation
            goal_type: Type of goal for Connector ("manhattan_distance" or "ratio_connected")
        """
        super().__init__(env)
        self._env: MarlEnv
        # Copy attributes
        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim
        self.goal_type = goal_type
        
        # Unpack correctly
        self.base_obs_dim = self._env.observation_spec.agents_view.shape[-1]
        self._goal_dim = goal_dim
        self.goal_start_idx = goal_start_idx
        self.goal_end_idx = goal_end_idx
        
        # After we append goal, total obs size will be:
        self.obs_dim = obs_dim + goal_dim
        # Select goal extraction function
        if "smax" in env_name.lower():
            self._extract_goal = self._smax_goal_extraction
        elif "connector" in env_name.lower():
            # Detect if VectorConnector or regular Connector
            if "vector" in env_name.lower():
                # VectorConnector uses same goal extraction logic
                if goal_type == "manhattan_distance":
                    self._extract_goal = self._vector_connector_goal_manhattan_distance
                elif goal_type == "manhattan_distance_per_agent":
                    self._extract_goal = self._vector_connector_goal_manhattan_distance_per_agent
                elif goal_type == "ratio_connected":
                    self._extract_goal = self._vector_connector_goal_ratio_connected
                else:
                    raise ValueError(f"Unknown goal_type for VectorConnector: {goal_type}. "
                                   f"Choose 'manhattan_distance', 'manhattan_distance_per_agent', or 'ratio_connected'")
            else:
                # Regular Connector
                if goal_type == "manhattan_distance":
                    self._extract_goal = self._connector_goal_manhattan_distance
                elif goal_type == "ratio_connected":
                    self._extract_goal = self._connector_goal_ratio_connected
                else:
                    raise ValueError(f"Unknown goal_type for Connector: {goal_type}. "
                                   f"Choose 'manhattan_distance' or 'ratio_connected'")
        else:
            raise ValueError(f"ICRL not implemented for {env_name}")
        
    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset and append goal to observation."""
        # Call wrapped environment
        state, timestep = self._env.reset(key)
        
        # Extract goal and append to observation
        timestep = self._append_goal_to_obs(state, timestep)
        
        return state, timestep
    
    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step and append goal to observation."""
        # Call wrapped environment
        state, timestep = self._env.step(state, action)
        
        # Extract goal and append to observation
        timestep = self._append_goal_to_obs(state, timestep)
        return state, timestep
    
    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Observation spec with goal dimension added."""
        base_spec = self._env.observation_spec
         
        # The observation is an Observation NamedTuple with agents_view
        # We need to update the agents_view spec
        old_agents_view_spec = base_spec.agents_view
        old_shape = old_agents_view_spec.shape
        
        # Add goal_dim to the last dimension
        # Convert to Python ints to ensure proper shape tuple
        new_shape = (int(old_shape[0]), int(old_shape[1]) + self._goal_dim)
        
        new_agents_view_spec = specs.BoundedArray(
            shape=new_shape,
            dtype=old_agents_view_spec.dtype,
            minimum=old_agents_view_spec.minimum,
            maximum=old_agents_view_spec.maximum,
        )
        
        # Replace agents_view spec in the observation spec
        return base_spec._replace(agents_view=new_agents_view_spec)

    
    def _append_goal_to_obs(
        self, 
        state: State, 
        timestep: TimeStep
    ) -> TimeStep:
        """Extract goal from state and append to observation."""
        # Extract goal based on environment type (pass timestep for extras access)
        goal = self._extract_goal(state, timestep.observation, timestep.extras)
        
        # Append goal to observation
        # timestep.observation.agents_view shape: (num_agents, obs_dim)
        # goal shape: (num_agents, goal_dim)
        obs = timestep.observation
        new_agents_view = jnp.concatenate([obs.agents_view[:, :self.base_obs_dim], goal], axis=-1)
        
        # Create new observation with goal appended
        new_obs = obs._replace(agents_view=new_agents_view)
        timestep = timestep.replace(observation=new_obs)
        
        return timestep


    @cached_property
    def obs_size(self) -> int:
        """Total observation size after goal is appended."""
        return self.obs_dim
    # =====================================================================
    # MPE Tag Goal Extraction (exact replication from mpe_tag_facmac_6a.py)
    # =====================================================================
    
    def _mpe_goal_extraction(
        self, 
        state: State, 
        obs: Union[Observation, ObservationGlobalState]
    ) -> chex.Array:
        """Extract goal for MPE Tag: minimum distance to nearest visible prey.
        
        This replicates the exact logic from envs/mpe_tag_facmac_6a.py lines 14-33.
        
        Args:
            state: JaxMarlState containing MPEState
            obs: Observation with agents_view
            
        Returns:
            Array of shape (num_agents, 2) with [min_dist, 0] for each agent
        """
        # Access the MPE state from JaxMarlState
        jaxmarl_state: JaxMarlState = state
        mpe_state: MPEState = jaxmarl_state.state
        
        # Get wrapped environment (should be MPEWrapper which wraps SimpleFacmacMPE)
        mpe_env = self._env._env  # MPEWrapper._env is the raw JaxMarl env
        
        # Number of adversaries (predators) and landmarks
        num_adversaries = mpe_env.num_agents - mpe_env.num_good
        num_landmarks = len(mpe_env.world.landmarks)
        
        # Extract agent observations (only adversaries)
        adv_obs = obs.agents_view[:num_adversaries]  # shape: (num_adversaries, obs_dim)
        
        def get_agent_min(agent_obs: chex.Array) -> chex.Array:
            """Compute minimum distance to nearest visible prey for one agent.
            
            This is the exact logic from mpe_tag_facmac_6a.py lines 15-28.
            """
            # Extract positions of other adversaries and good agents from observation
            other_start = 4 + 2 * num_landmarks
            
            # Other adversaries' positions
            adv_pos = agent_obs[other_start:other_start + 2*(num_adversaries-1)]
            adv_pos = jnp.concatenate((jnp.zeros(2), adv_pos))  # Add self (zeros)
            adv_pos = jnp.reshape(adv_pos, (-1, 2))
            
            # Good agents' positions (prey)
            good_pos = agent_obs[other_start + 2*(num_adversaries-1):
                                other_start + 2*(mpe_env.num_agents-1)]
            good_pos = jnp.reshape(good_pos, (-1, 2))
            
            # Compute distance matrix: (adversaries, prey)
            dist_mat = (adv_pos[:, None, :] - good_pos[None, :, :]) ** 2
            dist_mat = jnp.sum(dist_mat, axis=2)
            
            # Get absolute good agent positions from state
            abs_good_pos = mpe_state.p_pos[num_adversaries:mpe_env.num_agents]
            
            # Check if prey is hidden (observed pos != actual pos - agent pos)
            good_hidden = jnp.sum((abs_good_pos - good_pos - agent_obs[2:4])**2, 
                                axis=1) > 0.01
            
            # Set hidden prey distances to very large value
            dist_mat = jnp.where(good_hidden, 99999999.0, dist_mat)
            
            # Return minimum distance
            return jnp.min(dist_mat)
        
        # Compute min distance for each adversary
        adv_min = jax.vmap(get_agent_min, in_axes=0)(adv_obs)
        
        # Return shape: (num_agents, 2) with [min_dist, 0]
        return jnp.concatenate((adv_min[:, None], jnp.zeros((num_adversaries, 1))), axis=1)
    
    # =====================================================================
    # SMAX Goal Extraction (exact replication from smax.py)
    # =====================================================================
    
    def _smax_goal_extraction(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract goal for SMAX: sum of visible enemy healths for each agent.
        
        This replicates the exact logic from envs/smax.py lines 19-38.
        CRITICAL FIX: When battle is won, return [0.0, 0.0] to match CRL behavior.
        
        Args:
            state: JaxMarlState containing SMAXState
            obs: Observation with agents_view
            
        Returns:
            Array of shape (num_agents, 2) with [sum_enemy_health, 0] for each agent
            OR [0.0, 0.0] when battle is won (matching CRL line 138: obs[:,-2] = 0.0)
        """
        # Access the SMAX state from JaxMarlState
        jaxmarl_state: JaxMarlState = state
        smax_state: SMAXState = jaxmarl_state.state
        
        # CRITICAL FIX: Check if battle is won - if so, return [0.0, 0.0] for all agents
        # This matches CRL's behavior: obs[:,-2] = 0.0 when won_battle (line 138 in crl/envs/smax.py)
        # JaxMarlState always has won_battle field (defaults to False if not set)
        won_battle = jaxmarl_state.won_battle
        
        # If battle is won, enemy health should be 0 (goal achieved)
        # Return [0.0, 0.0] for all agents to match CRL's observation modification
        def return_zero_goal():
            return jnp.zeros((self.num_agents, 2))
        
        def compute_goal():
            # Get wrapped environment (should be SmaxWrapper)
            smax_env = self._env._env  # SmaxWrapper._env is the raw JaxMarl env
            # Extract agent observations
            ally_obs = obs.agents_view  # shape: (num_agents, obs_dim)
            
            def get_agent_enemy_healths(agent_obs: chex.Array) -> chex.Array:
                """Compute sum of visible enemy healths for one agent.
                
                This is the exact logic from smax.py lines 20-30.
                """
                # Extract enemy features from observation
                num_feat = len(smax_env.unit_features)
                start_idx = (smax_env.num_agents - 1) * num_feat
                end_idx = start_idx + smax_env.num_enemies * num_feat
                enemy_features = agent_obs[start_idx:end_idx]
                enemy_features = jnp.reshape(enemy_features, (smax_env.num_enemies, -1))
                
                # First feature is health (normalized)
                enemy_healths = enemy_features[:, 0]
                
                # Get actual enemy healths from state
                max_healths = smax_env.unit_type_health[
                    smax_state.state.unit_types[jnp.arange(smax_env.num_enemies) + smax_env.num_agents]
                ]
                enemy_healths_real = smax_state.state.unit_health[
                    smax_env.num_agents:smax_env.num_agents + smax_env.num_enemies
                ]
                enemy_healths_real = enemy_healths_real / max_healths
                
                # Check if observed health matches actual (within tolerance)
                # If not, enemy is not visible → set to 1.0 (full health)
                enemy_healths_obs = jnp.where(
                    (enemy_healths - enemy_healths_real)**2 < 0.01,
                    enemy_healths,
                    1.0
                )
                
                # Special case: if agent is dead (all obs = 0), use real healths
                enemy_healths_obs = jnp.where(
                    jnp.all(agent_obs == 0),
                    enemy_healths_real,
                    enemy_healths_obs
                )
                
                # Return sum of visible enemy healths
                return jnp.sum(enemy_healths_obs)
            
            # Compute sum of enemy healths for each agent
            ally_enemy_healths = jax.vmap(get_agent_enemy_healths)(ally_obs)
            
            # Return (num_agents, 2) with [sum_health, 0] to match observation format
            return jnp.concatenate((ally_enemy_healths[:, None], 
                                   jnp.zeros((self.num_agents, 1))), axis=1)
        
        # Use conditional to return zero goal when battle is won
        return jax.lax.cond(
            won_battle,
            return_zero_goal,
            compute_goal
        )
    
    # =====================================================================
    # Connector Goal Extraction
    # =====================================================================
    
    def _connector_goal_manhattan_distance(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract goal for Connector: total Manhattan distance to all targets.
        
        Goal Design (matching SMAX structure):
        - Dimension 0: Current total Manhattan distance (gets replaced during training)
        - Dimension 1: Ultimate goal (0.0 = all agents connected, constant signal)
        
        Args:
            state: ConnectorMarlState (wraps ConnectorState) from ConnectorWrapper
            obs: Observation with agents_view
            
        Returns:
            Array of shape (num_agents, 2) with [total_distance, 0.0] for each agent
            """
        # Extract ConnectorState from ConnectorMarlState wrapper
        connector_state: ConnectorState = state.state
        
        # Direct access to agent positions and targets (simpler and faster than grid search)
        positions = connector_state.agents.position  # Shape: (num_agents, 2)
        targets = connector_state.agents.target      # Shape: (num_agents, 2)
        
        # Compute Manhattan distance for each agent
        distances = jnp.sum(jnp.abs(positions - targets), axis=1)  # Shape: (num_agents,)
        
        # Total distance (team goal) - sum across all agents
        total_distance = jnp.sum(distances)  # Shape: scalar
        
        # Normalize goal to [0, 1] range for ICRL
        # Maximum possible distance for grid_size N and num_agents M:
        #   Max Manhattan distance per agent: 2*(N-1) (corner to corner)
        #   Max total distance: M * 2*(N-1)
        # For 10x10 grid with 10 agents: max = 10 * 2*9 = 180
        grid_size = connector_state.grid.shape[0]
        max_distance_per_agent = 2 * (grid_size - 1)  # Corner to corner: (N-1) + (N-1)
        max_total_distance = self.num_agents * max_distance_per_agent
        normalized_distance = total_distance / max_total_distance
        
        # Clamp to 0 when very close
        normalized_distance = jnp.where(normalized_distance < 0.001, 0.0, normalized_distance)
        
        # Return 2D goal like SMAX (normalized to [0, 1])
        # Dimension 0: current distance [0-1] (dynamic, gets replaced during hindsight relabeling)
        # Dimension 1: ultimate goal = 0.0 (constant, tells actor what to achieve)
        # Each agent gets the SAME team goal (normalized_distance)
        current_distance = jnp.full((self.num_agents, 1), normalized_distance, dtype=jnp.float32)
        ultimate_goal = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        
        # Shape: (num_agents, 2)
        return jnp.concatenate([current_distance, ultimate_goal], axis=1)
    
    def _connector_goal_ratio_connected(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract goal for Connector: ratio of connected agents (from env extras).
        
        Goal Design (matching SMAX structure):
        - Dimension 0: Remaining ratio to connect (1.0 - ratio_connections)
        - Dimension 1: Ultimate goal (0.0 = all agents connected, constant signal)
        
        This goal type uses the `ratio_connections` metric already computed by Jumanji,
        which represents the fraction of agents that have reached their targets.
        
        Goal semantics:
        - Start: ratio_connections = 0.0 → goal = 1.0 (100% remaining)
        - End:   ratio_connections = 1.0 → goal = 0.0 (0% remaining, all connected)
        
        Args:
            state: ConnectorMarlState (wraps ConnectorState) from ConnectorWrapper
            obs: Observation with agents_view
            extras: Timestep extras containing 'ratio_connections' metric
            
        Returns:
            Array of shape (num_agents, 2) with [remaining_ratio, 0.0] for each agent
        """
        # 🔍 Get ratio_connections from extras (computed by Jumanji Connector and passed through wrapper)
        # extras contains env_metrics which has ratio_connections
        env_metrics = extras.get("env_metrics", {})
        ratio_connected = env_metrics.get("ratio_connections", 0.0)
        
        # Goal: remaining ratio to achieve (starts high, decreases to 0)
        # When ratio_connected = 0.0 (no connections) → remaining_ratio = 1.0
        # When ratio_connected = 1.0 (all connected) → remaining_ratio = 0.0
        remaining_ratio = 1.0 - ratio_connected
        
        # Clamp to 0 when fully connected
        remaining_ratio = jnp.where(remaining_ratio < 0.001, 0.0, remaining_ratio)
        
        # ✨ Return 2D goal like SMAX (already normalized in [0, 1]!)
        # Dimension 0: remaining ratio [0-1] (dynamic, gets replaced during hindsight relabeling)
        # Dimension 1: ultimate goal = 0.0 (constant, tells actor what to achieve)
        # 
        # CRITICAL: Reshape correctly for each agent
        # Each agent gets the SAME team goal (remaining_ratio)
        current_goal = jnp.full((self.num_agents, 1), remaining_ratio, dtype=jnp.float32)
        ultimate_goal = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        
        # Shape: (num_agents, 2)
        return jnp.concatenate([current_goal, ultimate_goal], axis=1)
    
    # =====================================================================
    # VectorConnector Goal Extraction
    # =====================================================================
    
    def _vector_connector_goal_manhattan_distance(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract goal for VectorConnector: total Manhattan distance to all targets.
        
        This is identical to ConnectorWrapper's manhattan distance goal, but handles
        VectorConnectorMarlState wrapper.
        
        Goal Design (matching SMAX structure):
        - Dimension 0: Current total Manhattan distance (gets replaced during training)
        - Dimension 1: Ultimate goal (0.0 = all agents connected, constant signal)
        
        Args:
            state: VectorConnectorMarlState (wraps ConnectorState) from VectorConnectorWrapper
            obs: Observation with agents_view (partial FOV observations)
            extras: Timestep extras
            
        Returns:
            Array of shape (num_agents, 2) with [total_distance, 0.0] for each agent
        """
        # Extract ConnectorState from VectorConnectorMarlState wrapper
        # VectorConnectorWrapper wraps state just like ConnectorWrapper does
        if hasattr(state, 'state'):
            connector_state: ConnectorState = state.state
        else:
            # Fallback for raw ConnectorState (shouldn't happen in practice)
            connector_state: ConnectorState = state
        
        # Direct access to agent positions and targets from state
        # NOTE: Even though agents have partial observability (FOV=2), the STATE
        # still contains full ground truth positions and targets!
        positions = connector_state.agents.position  # Shape: (num_agents, 2)
        targets = connector_state.agents.target      # Shape: (num_agents, 2)
        
        # Compute Manhattan distance for each agent
        distances = jnp.sum(jnp.abs(positions - targets), axis=1)  # Shape: (num_agents,)
        
        # Total distance (team goal) - sum across all agents
        total_distance = jnp.sum(distances)  # Shape: scalar
        
        # Normalize goal to [0, 1] range for ICRL
        grid_size = connector_state.grid.shape[0]
        max_distance_per_agent = 2 * (grid_size - 1)  # Corner to corner
        max_total_distance = self.num_agents * max_distance_per_agent
        normalized_distance = total_distance / max_total_distance
        
        # Clamp to 0 when very close
        normalized_distance = jnp.where(normalized_distance < 0.001, 0.0, normalized_distance)
        
        # Return 2D goal like SMAX (normalized to [0, 1])
        # Dimension 0: current distance [0-1] (dynamic, gets replaced during hindsight relabeling)
        # Dimension 1: ultimate goal = 0.0 (constant, tells actor what to achieve)
        # Each agent gets the SAME team goal (normalized_distance)
        current_distance = jnp.full((self.num_agents, 1), normalized_distance, dtype=jnp.float32)
        ultimate_goal = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        
        # Shape: (num_agents, 2)
        return jnp.concatenate([current_distance, ultimate_goal], axis=1)
    
    def _vector_connector_goal_manhattan_distance_per_agent(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract PER-AGENT goal for VectorConnector: each agent's own Manhattan distance.
        
        This is the FIXED version that solves the scaling problem!
        
        Key Differences from team-level version:
        - Each agent gets ITS OWN distance as goal (not team total)
        - Scales naturally: each agent's goal ∈ [0, 1] regardless of team size
        - Clear credit assignment: agent sees when ITS actions help
        - Matches SMAX's per-agent goal design
        
        Goal Design (matching SMAX structure):
        - Dimension 0: Agent's OWN Manhattan distance to its target [0-1]
        - Dimension 1: Ultimate goal (0.0 = agent connected, constant signal)
        
        Signal Propagation (how each agent learns from its goal):
        - Agent i observes: obs[i] (shape: obs_dim)
        - Agent i gets goal: goal[i] = [my_distance, 0.0] (shape: 2)
        - Agent i's observation becomes: [obs[i], goal[i]] (shape: obs_dim + 2)
        - Policy network: π(action | obs[i], goal[i]) - ONLY sees its own obs+goal!
        - Critic network: Q(obs[i], action[i], goal[i]) - ONLY sees its own obs+action+goal!
        
        No mixing occurs because:
        - Each agent's network input is separate: agent_0 never sees agent_1's goal
        - Parameter sharing means same network weights, but different inputs
        - Like: same_function(agent_0_data) vs same_function(agent_1_data)
        
        Args:
            state: VectorConnectorMarlState (wraps ConnectorState) from VectorConnectorWrapper
            obs: Observation with agents_view (partial FOV observations)
            extras: Timestep extras
            
        Returns:
            Array of shape (num_agents, 2) where:
            - goal[i, 0] = agent i's distance to its target (normalized [0-1])
            - goal[i, 1] = 0.0 (ultimate goal for all agents)
        """
        # Extract ConnectorState from VectorConnectorMarlState wrapper
        if hasattr(state, 'state'):
            connector_state: ConnectorState = state.state
        else:
            connector_state: ConnectorState = state
        
        # Direct access to agent positions and targets from state
        positions = connector_state.agents.position  # Shape: (num_agents, 2)
        targets = connector_state.agents.target      # Shape: (num_agents, 2)
        
        # Compute Manhattan distance for EACH agent (NOT summed!)
        distances = jnp.sum(jnp.abs(positions - targets), axis=1)  # Shape: (num_agents,)
        
        # Normalize PER AGENT (not team total!)
        grid_size = connector_state.grid.shape[0]
        max_distance_per_agent = 2 * (grid_size - 1)  # Corner to corner
        normalized_distances = distances / max_distance_per_agent  # Shape: (num_agents,)
        
        # Clamp to 0 when very close (per agent)
        normalized_distances = jnp.where(normalized_distances < 0.001, 0.0, normalized_distances)
        
        # Return 2D goal like SMAX (normalized to [0, 1])
        # Dimension 0: EACH agent's OWN distance [0-1] (different for each agent!)
        # Dimension 1: ultimate goal = 0.0 (constant, tells actor what to achieve)
        current_distance = normalized_distances[:, None]  # Shape: (num_agents, 1) - DIFFERENT per agent!
        ultimate_goal = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        
        # Shape: (num_agents, 2)
        # Example output for 5 agents:
        # [[0.42, 0.0],  # Agent 0: 42% of max distance from its target
        #  [0.83, 0.0],  # Agent 1: 83% of max distance from its target
        #  [0.17, 0.0],  # Agent 2: 17% of max distance from its target
        #  [1.00, 0.0],  # Agent 3: 100% of max distance (far from target)
        #  [0.25, 0.0]]  # Agent 4: 25% of max distance from its target
        return jnp.concatenate([current_distance, ultimate_goal], axis=1)
    
    def _vector_connector_goal_ratio_connected(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract goal for VectorConnector: ratio of agents NOT yet connected.
        
        This goal uses the ratio_connections metric from Jumanji's Connector environment,
        which is passed through VectorConnectorWrapper's extras.
        
        Goal Design (matching SMAX structure):
        - Dimension 0: Remaining ratio to connect (1.0 - ratio_connections)
        - Dimension 1: Ultimate goal (0.0 = all agents connected, constant signal)
        
        Goal semantics:
        - Start: ratio_connections = 0.0 → goal = 1.0 (100% remaining)
        - End:   ratio_connections = 1.0 → goal = 0.0 (0% remaining, all connected)
        
        Args:
            state: VectorConnectorMarlState (wraps ConnectorState) from VectorConnectorWrapper
            obs: Observation with agents_view (partial FOV observations)
            extras: Timestep extras containing 'ratio_connections' metric
            
        Returns:
            Array of shape (num_agents, 2) with [remaining_ratio, 0.0] for each agent
        """
        # Get ratio_connections from extras
        # VectorConnectorWrapper passes timestep.extras through to env_metrics (line 539 in jumanji.py)
        # Jumanji's Connector computes ratio_connections = jnp.mean(state.agents.connected)
        env_metrics = extras.get("env_metrics", {})
        ratio_connected = env_metrics.get("ratio_connections", 0.0)
        
        # Goal: remaining ratio to achieve (starts high, decreases to 0)
        remaining_ratio = 1.0 - ratio_connected
        
        # Clamp to 0 when fully connected
        remaining_ratio = jnp.where(remaining_ratio < 0.001, 0.0, remaining_ratio)
        
        # Return 2D goal like SMAX (already normalized in [0, 1]!)
        # Each agent gets the SAME team goal (remaining_ratio)
        current_goal = jnp.full((self.num_agents, 1), remaining_ratio, dtype=jnp.float32)
        ultimate_goal = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        
        # Shape: (num_agents, 2)
        return jnp.concatenate([current_goal, ultimate_goal], axis=1)
    
    def _vector_connector_goal_ratio_connected_from_state(
        self,
        state: State,
        obs: Union[Observation, ObservationGlobalState],
        extras: Dict
    ) -> chex.Array:
        """Extract goal for VectorConnector: ratio of agents NOT yet connected (from state).
        
        This is an alternative to _vector_connector_goal_ratio_connected that computes
        the ratio directly from state.agents.connected instead of using extras.
        
        This is useful if you want to avoid dependency on extras or want more direct control.
        
        Goal Design (matching SMAX structure):
        - Dimension 0: Remaining ratio to connect (1.0 - ratio_connections)
        - Dimension 1: Ultimate goal (0.0 = all agents connected, constant signal)
        
        Args:
            state: VectorConnectorMarlState (wraps ConnectorState) from VectorConnectorWrapper
            obs: Observation with agents_view (partial FOV observations)
            extras: Timestep extras (not used in this version)
            
        Returns:
            Array of shape (num_agents, 2) with [remaining_ratio, 0.0] for each agent
        """
        # Extract ConnectorState from wrapper
        if hasattr(state, 'state'):
            connector_state: ConnectorState = state.state
        else:
            connector_state: ConnectorState = state
        
        # Get connection status directly from state
        # state.agents.connected is a boolean array of shape (num_agents,)
        # Each element is True if that agent has reached its target
        connected_status = connector_state.agents.connected  # Shape: (num_agents,)
        
        # Compute ratio of connected agents
        # This is equivalent to Jumanji's: jnp.mean(state.agents.connected)
        ratio_connected = jnp.mean(connected_status.astype(jnp.float32))
        
        # Goal: remaining ratio to achieve (starts high, decreases to 0)
        remaining_ratio = 1.0 - ratio_connected
        
        # Clamp to 0 when fully connected
        remaining_ratio = jnp.where(remaining_ratio < 0.001, 0.0, remaining_ratio)
        
        # Return 2D goal like SMAX
        current_goal = jnp.full((self.num_agents, 1), remaining_ratio, dtype=jnp.float32)
        ultimate_goal = jnp.zeros((self.num_agents, 1), dtype=jnp.float32)
        
        # Shape: (num_agents, 2)
        return jnp.concatenate([current_goal, ultimate_goal], axis=1)