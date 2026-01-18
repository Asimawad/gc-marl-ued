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

from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import variance_scaling, orthogonal
import tensorflow_probability.substrates.jax.distributions as tfd

from mava.networks.distributions import IdentityTransformation


class SAEncoder(nn.Module):
    """State-Action Encoder for ICRL - encodes (s,a) pairs using a configurable torso.
    
    Used for continuous actions where action is concatenated with state.
    """
    
    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            s: State (observation without goal)
            a: Action (continuous or one-hot)
            
        Returns:
            output_dim-dimensional encoding
        """
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        # Concatenate state and action
        x = jnp.concatenate([s, a], axis=-1)
        
        # Use configurable torso instead of hardcoded layers
        x = self.torso(x)
        
        # Output layer: configurable dimension
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class DiscreteSAEncoder(nn.Module):
    """State-Action Encoder for discrete ICRL.
    
    Outputs embeddings for ALL actions at once: [batch, num_actions, emb_dim]
    This allows computing Q-values for all actions efficiently in discrete SAC.
    """
    
    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    num_actions: int
    emb_dim: int = 64
    
    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        """Forward pass.
        
        Args:
            s: State (observation without goal)
            
        Returns:
            [batch, num_actions, emb_dim] - embedding for each action
        """
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        # Use configurable torso
        x = self.torso(s)
        
        # Output layer: emb_dim for each action
        x = nn.Dense(
            self.emb_dim * self.num_actions,
            kernel_init=lecun_uniform,
            bias_init=bias_init,
        )(x)
        
        # Reshape to [batch..., num_actions, emb_dim]
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.num_actions, self.emb_dim)
        
        return x


class GoalEncoder(nn.Module):
    """Goal Encoder for ICRL - encodes goals using a configurable torso."""
    
    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    emb_dim: int = 64
    
    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            g: Goal
            
        Returns:
            emb_dim-dimensional encoding
        """
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        # Use configurable torso
        x = self.torso(g)
        
        # Output layer: configurable dimension
        x = nn.Dense(self.emb_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class ICRLActor(nn.Module):
    """Actor network for ICRL - outputs continuous action distribution using a configurable torso."""
    
    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    action_size: int
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.
        
        Args:
            x: Observation (base obs + goal concatenated)
            
        Returns:
            (mean, log_std) tuple for action distribution
        """
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        # Use configurable torso
        x = self.torso(x)
        
        # Two output heads for mean and log_std
        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        
        # Clip log_std to reasonable range
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std


class DiscreteICRLActor(nn.Module):
    """Actor network for discrete ICRL - outputs categorical distribution over actions."""
    
    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    action_size: int
    
    @nn.compact
    def __call__(self, x: jax.Array, action_mask: jax.Array) -> tfd.TransformedDistribution:
        """Forward pass.
        
        Args:
            x: Observation (base obs + goal concatenated)
            action_mask: Boolean mask for available actions
            
        Returns:
            Categorical distribution over actions (with masking)
        """
        # Use configurable torso
        x = self.torso(x)
        
        # Output logits for each action
        actor_logits = nn.Dense(self.action_size, kernel_init=orthogonal(0.01))(x)
        
        # Apply action masking
        masked_logits = jnp.where(
            action_mask,
            actor_logits,
            jnp.finfo(jnp.float32).min,
        )
        
        # Return categorical distribution with identity transformation (for API consistency)
        return IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
