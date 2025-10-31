from typing import Dict, Tuple
import jax
import jax.numpy as jnp
import chex
from brax import envs
from jaxmarl.environments.mabrax.mabrax_env import MABraxEnv
from functools import partial

class PositionBasedReward(MABraxEnv):
    """Goal-conditioned version of MABraxEnv with sparse rewards."""
    
    def __init__(
        self,
        env_name: str,
        goal_pos: Tuple[float, float, float] = (5.0, 0.0, 0.0),
        threshold: float = 0.5,
        sparse = True,
        **kwargs
    ):
        """
        Args:
            env_name: Name of the base environment
            goal_pos: Target (x,y,z) coordinate to reach
            threshold: Distance threshold to consider goal reached
            **kwargs: Additional arguments passed to MABraxEnv
        """
        super().__init__(env_name, **kwargs)
        self.goal_pos = jnp.array(goal_pos)[:2]
        self.threshold = threshold
        self.sparse = sparse

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], envs.State]:
        state = self.env.reset(key)
        state.info.update({"success": 0.0, "curr_success": 0.0})
        return self.get_obs(state), state
        
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: envs.State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], envs.State, Dict[str, float], Dict[str, bool], Dict]:
        # Get next state from parent class
        global_action = self.map_agents_to_global_action(actions)
        next_state = self.env.step(state, global_action)
        
        # Extract agent position from state
        # Note: This assumes position is in the first 3 dimensions of qp.pos
        # You may need to adjust this based on your specific environment
        agent_pos = next_state.pipeline_state.q[:2]
        
        # Calculate distance to goal
        distance = jnp.linalg.norm(agent_pos - self.goal_pos)
        
        # Sparse reward: 1.0 if within threshold, 0.0 otherwise
        sparse_reward = jnp.where(distance < self.threshold, 1.0, 0.0)
        dense_reward = -1.0*distance
        reward = sparse_reward if self.sparse else dense_reward

        # track success
        success = jnp.where(next_state.done, state.info["curr_success"], state.info["success"])
        success_curr = jnp.where(next_state.done, 0.0, state.info["curr_success"]+sparse_reward)
        next_state.info.update({"success": success, "curr_success": success_curr})

        # Update reward in state
        next_state = next_state.replace(reward=reward)
        
        # Create observation and reward dictionaries
        observations = self.get_obs(next_state)
        rewards = {agent: reward for agent in self.agents}
        rewards["__all__"] = reward
        dones = {agent: next_state.done.astype(jnp.bool_) for agent in self.agents}
        dones["__all__"] = next_state.done.astype(jnp.bool_)
        
        return observations, next_state, rewards, dones, next_state.info
        
# Example usage for different environments
class PositionAnt(PositionBasedReward):
    def __init__(self, goal_pos=(5.0, 0.0, 0.0), threshold=0.5, **kwargs):
        super().__init__("ant_4x2", goal_pos=goal_pos, threshold=threshold, **kwargs)
class PositionWalker(PositionBasedReward):
    def __init__(self, goal_pos=(5.0, 0.0, 0.0), threshold=0.5, **kwargs):
        super().__init__("walker2d_2x3", goal_pos=goal_pos, threshold=threshold, **kwargs)
