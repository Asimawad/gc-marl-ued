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

from mava.types import Observation, ObservationGlobalState, State


# Type aliases
Metrics: TypeAlias = Dict[str, Array]
Networks: TypeAlias = Tuple[nn.Module, nn.Module, nn.Module]  # (actor, sa_encoder, goal_encoder)
Optimisers: TypeAlias = Tuple[
    optax.GradientTransformation,  # actor optimizer
    optax.GradientTransformation,  # critic (both encoders) optimizer
    optax.GradientTransformation   # alpha optimizer
]


class ICRLParams(NamedTuple):
    """Parameters for ICRL networks."""
    actor: FrozenVariableDict
    sa_encoder: FrozenVariableDict
    goal_encoder: FrozenVariableDict
    log_alpha: Array




class OptStates(NamedTuple):
    """Optimizer states for ICRL."""
    actor: optax.OptState
    critic: optax.OptState  # Both encoders share one optimizer
    alpha: optax.OptState


class Transition(NamedTuple):
    """Transition for ICRL - includes goal information."""
    observation: Observation  # Observation WITH goal appended
    action: Array
    reward: Array
    discount: Array
    avail_actions: Array  # Available actions mask for discrete action spaces
    extras: Dict[str, Any]


BufferState: TypeAlias = TrajectoryBufferState[Transition]



class LearnerState(NamedTuple):
    """Complete learner state for ICRL (restructured version)."""
    params: ICRLParams
    opt_states: OptStates
    buffer_state: BufferState
    key: PRNGKey
    env_state: State
    last_timestep: Any  # TimeStep from environment
    

# class LearnerState(NamedTuple):
#     """Complete learner state for ICRL (restructured version)."""
#     params: ICRLParams
#     opt_states: OptStates
#     buffer_state: BufferState
#     key: PRNGKey
#     env_state: Any  # Brax State (no separate timestep)