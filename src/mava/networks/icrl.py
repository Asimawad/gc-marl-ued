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

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import variance_scaling


class SAEncoder(nn.Module):
    """State-Action Encoder for ICRL - encodes (s,a) pairs using a configurable torso."""

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    output_dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            s: State (observation without goal)
            a: Action

        Returns:
            output_dim-dimensional encoding
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Concatenate state and action
        x = jnp.concatenate([s, a], axis=-1)

        # Use configurable torso instead of hardcoded layers
        x = self.torso(x)

        # Output layer: configurable dimension
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class GoalEncoder(nn.Module):
    """Goal Encoder for ICRL - encodes goals using a configurable torso."""

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    output_dim: int = 64

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            g: Goal

        Returns:
            output_dim-dimensional encoding
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Use configurable torso
        x = self.torso(g)

        # Output layer: configurable dimension
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
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
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
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


class PQNStateActionEncoder(nn.Module):
    """
    State-Action Encoder for PQN-CRL - encodes (s,a) pairs using a configurable torso.
    """

    torso: nn.Module  # Configurable torso (e.g., MLPTorso)
    num_actions: int
    output_dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            s: State observation [batch, state_dim]

        Returns:
            Per-action representations [batch, num_actions, output_dim]
        """
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Process state through torso
        x = self.torso(s)

        # Output layer: num_actions * output_dim
        x = nn.Dense(self.num_actions * self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        # Reshape to [batch, num_actions, output_dim]
        batch_size = x.shape[:-1]
        x = x.reshape(*batch_size, self.num_actions, self.output_dim)

        return x


class SA_encoder(nn.Module):
    """
    State encoder that outputs representations for ALL actions at once.
    Output shape: (batch, action_size * rep_size)
    This allows implicit action selection via Q-value comparison.
    """
    action_size: int
    rep_size: int
    norm_type: str = "layer_norm"
    
    @nn.compact
    def __call__(self, s: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(s)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        # Output: (batch, action_size * rep_size)
        x = nn.Dense(self.action_size * self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class CRLExplicitActionEncoder(nn.Module):
    """State-Action Encoder with explicit action encoding for discrete CRL.
    
    Key difference from CRLStateActionEncoder: Action is concatenated at input
    and flows through ALL layers, giving action full representational power.
    
    This matches SAC's critic architecture where (s, a) are concatenated at input.
    Q-values are computed by calling this encoder once per action (vmapped).
    
    Input: state (obs_dim) + action_onehot (action_size)
    Output: Single representation (rep_size)
    """
    
    torso: nn.Module  # Configurable torso (e.g., MLPTorso with LayerNorm)
    rep_size: int = 64
    
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            s: State observation [..., state_dim]
            a: One-hot encoded action [..., action_size]
            
        Returns:
            Representation [..., rep_size]
        """
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        # Concatenate state and action - action flows through ALL layers
        x = jnp.concatenate([s, a], axis=-1)
        
        # Process through torso
        x = self.torso(x)
        
        # Output layer: single representation
        x = nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        
        return x




class small_G_encoder(nn.Module):
    rep_size: int
    norm_type = "layer_norm"
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(g)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x

class sa_ConvEncoder(nn.Module):
    output_size:  int
    norm_type = "layer_norm"
    
    @nn.compact
    def __call__(self, x):
        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(16, kernel_size=(2, 2), kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(32, kernel_size=(2, 2), kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(64, kernel_size=(2, 2), kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(256, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.output_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)        
        return x


class sa_ConvEncoder_ActionInput(nn.Module):
    """
    State-Action ConvEncoder that takes (grid_obs, action_onehot) as input.
    Matches the Brax SA_encoder approach: action is concatenated with CNN features
    and processed through MLP layers.
    
    This allows computing Q-values one action at a time using vmap.
    """
    rep_size: int
    norm_type: str = "layer_norm"
    
    @nn.compact
    def __call__(self, grid_obs: jnp.ndarray, action_onehot: jnp.ndarray):
        """Forward pass.
        
        Args:
            grid_obs: Grid observation with shape (batch, H, W, 1)
            action_onehot: One-hot encoded action with shape (batch, action_size)
            
        Returns:
            Representation with shape (batch, rep_size)
        """
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        # CNN to process grid observation
        x = nn.Conv(16, kernel_size=(2, 2), kernel_init=lecun_uniform, bias_init=bias_init)(grid_obs)
        x = nn.swish(x)
        x = nn.Conv(32, kernel_size=(2, 2), kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(64, kernel_size=(2, 2), kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = nn.swish(x)
        
        # Flatten CNN output
        x = x.reshape((x.shape[0], -1))
        
        # Concatenate with action - action flows through MLP layers (like Brax)
        x = jnp.concatenate([x, action_onehot], axis=-1)
        
        # MLP layers with LayerNorm (matching Brax SA_encoder)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        
        return x