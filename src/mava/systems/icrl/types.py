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
from typing import Any, Dict, Tuple

import flax.linen as nn
import optax
from chex import Array, PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from typing_extensions import NamedTuple, TypeAlias

from mava.types import Observation, State

# Type aliases
Metrics: TypeAlias = Dict[str, Array]
Networks: TypeAlias = Tuple[nn.Module, nn.Module, nn.Module]  # (actor, sa_encoder, goal_encoder)
Optimisers: TypeAlias = Tuple[
    optax.GradientTransformation,  # actor optimizer
    optax.GradientTransformation,  # critic (both encoders) optimizer
    optax.GradientTransformation,  # alpha optimizer
]


class ICRLParams(NamedTuple):
    """Parameters for ICRL networks."""

    actor: FrozenVariableDict = None
    sa_encoder: FrozenVariableDict = None
    goal_encoder: FrozenVariableDict = None
    log_alpha: Array = None
    # Target network params for Polyak averaging (optional, used in DQN/PQN-CRL)
    target_sa_encoder: FrozenVariableDict = None
    target_goal_encoder: FrozenVariableDict = None



class OptStates(NamedTuple):
    """Optimizer states for ICRL."""

    actor: optax.OptState = None
    critic: optax.OptState = None
    alpha: optax.OptState = None


class Transition(NamedTuple):
    """Transition for ICRL - includes goal information."""

    observation: Observation  # Observation WITH goal appended
    action: Array
    reward: Array
    discount: Array
    avail_actions: Array  # Available actions mask for discrete action spaces
    extras: Dict[str, Any]
    sampled_goal: Array = None  # [N_env, N_agent, 1]

class PQNTransition(NamedTuple):
    """Transition for PQN-CRL."""
    observation: Observation  # Observation WITH goal appended
    action: Array
    sampled_goal: Array = None  # [N_env, N_agent, 1]

BufferState: TypeAlias = TrajectoryBufferState[Transition]


class LearnerState(NamedTuple):
    """Complete learner state for ICRL (restructured version)."""

    params: ICRLParams
    opt_states: OptStates
    buffer_state: BufferState = None  # Buffer state
    key: PRNGKey = None  # Random key
    env_state: State = None  # Environment state
    last_timestep: Any = None  # TimeStep from environment
    t: Array = None  # Timestep counter for epsilon decay
    target_params: ICRLParams = None  # Target network params for stability 