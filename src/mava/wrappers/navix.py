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

import jax
import jax.numpy as jnp
import navix as nx
from typing import Dict, Any
from brax.envs.base import State

class NavixEnv:
    """Wrapper for Navix environments to match Brax-like interface."""

    def __init__(self, env_name: str = "gcrl_door_key"):
        """Initialize Navix environment wrapper.

        Args:
            env_name: Name of the registered Navix environment
        """
        self.env_name = env_name
        self._env = nx.make(env_name)
        self.num_agents = 1  # Navix environments are single-agent

        # Cache static values to avoid tracer issues
        shape = self._env.observation_space.shape
        # Grid observations only - no goal concatenation
        self._observation_size = int(jnp.prod(jnp.array(shape)))
        self._action_size = int(self._env.action_space.n)
        self._grid_shape = shape[:2]  # (H, W)
        
        # Create vmapped versions of reset and step for batch processing
        self._reset_single = jax.jit(self._reset_single_fn)
        self._step_single = jax.jit(self._step_single_fn)
        self.reset = jax.jit(jax.vmap(self._reset_single_fn))
        self.step = jax.jit(jax.vmap(self._step_single_fn))

    @property
    def observation_size(self) -> int:
        """Get observation size including goal coordinates."""
        return self._observation_size

    @property
    def action_size(self) -> int:
        """Get action space size."""
        return self._action_size

    def _reset_single_fn(self, key: jax.Array):
        """Reset a single environment instance.

        Args:
            key: JAX random key

        Returns:
            State with obs, reward, done, metrics, and info
        """
        

        # Reset Navix environment
        timestep = self._env.reset(key)

        # Return grid observation only (shape: H, W, 1)
        # Goal will be stored separately in actor_step from player position
        obs = timestep.observation[None, :]  # Add batch dim for single agent

        # Use the random key to generate a unique seed for episode identification in HER
        seed = jax.random.randint(key, (), 0, 2**31 - 1)

        # Create info dict
        info = {
            "seed": seed,
            "truncation": timestep.is_truncation(),
            "return": jnp.array(0.0),
        }

        # Create Brax-like state
        state = State(
            pipeline_state=timestep,
            obs=obs,
            reward=jnp.array(0.0),
            done=timestep.is_done(),
            metrics={},
            info=info,
        )

        return state

    def _step_single_fn(self, state, action: jax.Array):
        """Step a single environment instance.

        Args:
            state: Current state
            action: Action to take (scalar)

        Returns:
            New state
        """

        # Extract timestep from state
        timestep = state.pipeline_state

        # Navix expects scalar action
        action_scalar = jnp.squeeze(action)

        # Step environment
        new_timestep = self._env.step(timestep, action_scalar)

        # Return grid observation only (shape: H, W, 1)
        # Goal will be stored separately in actor_step from player position
        obs = new_timestep.observation[None, :]  # Add batch dim for single agent

        # Update info - use functional update to avoid dict mutation in JAX
        new_info = {
            "truncation": new_timestep.is_truncation(),
            "return": state.info["return"] + new_timestep.reward,
            "seed": state.info["seed"],
        }

        # Create new state
        new_state = State(
            pipeline_state=new_timestep,
            obs=obs,
            reward=new_timestep.reward,
            done=new_timestep.is_done(),
            metrics={},
            info=new_info,
        )

        return new_state

    def get_avail_actions(self, state):
        """Get available actions (all actions available in Navix)."""
        # In Navix, all actions are always available
        # Use cached _action_size to avoid tracer issues
        return jnp.ones((self.num_agents, self._action_size))
