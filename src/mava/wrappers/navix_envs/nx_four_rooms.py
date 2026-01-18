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
import navix as nx
import jax.numpy as jnp

from jax import Array
from flax import struct
from typing import Union

from navix.environments import Environment
from navix.states import State
from navix import register_env
from navix.components import EMPTY_POCKET_ID, Directional, HasColour, Openable
from navix.rendering.cache import RenderingCache
from navix.entities import Entities, Player, Wall, EntityIds
from navix.grid import random_positions, random_directions, room, horizontal_wall, vertical_wall
from navix.spaces import Discrete

four_rooms_goal_coords = jnp.array([14, 14], dtype=jnp.int32)


def free(prev_state: State, action: Array, state: State) -> Array:
    return jnp.zeros_like(nx.events.on_ball_hit(state), dtype=jnp.bool_)


def terminate_on_goal(prev_state: State, action: Array, state: State) -> Array:
    """Terminate when agent reaches the goal [14, 14]."""
    at_goal = jnp.all(state.entities["player"].position == jnp.array([14, 14], dtype=jnp.int32))
    return jnp.asarray(at_goal, dtype=jnp.bool_)


class gcrl_FourRooms(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None):
        assert self.height > 4, f"Insufficient height for room {self.height} < 4"
        assert self.width > 4, f"Insufficient width for room {self.width} < 4"

        key, k1, k2 = jax.random.split(key, 3)

        # Map
        grid = room(height=self.height, width=self.width)

        # Vertical partition (fixed for easier learning)
        opening_1 = 4
        opening_2 = 12
        openings = jnp.stack([opening_1, opening_2])
        wall_pos_vert = vertical_wall(grid, self.width // 2, openings)

        # Horizontal partition (fixed for easier learning)
        opening_1 = 4
        opening_2 = 12
        openings = jnp.stack([opening_1, opening_2])
        wall_pos_hor = horizontal_wall(grid, self.height // 2, openings)

        walls_pos = jnp.concatenate([wall_pos_vert, wall_pos_hor])
        walls = Wall.create(position=walls_pos)

        # Player starts at fixed position [1,1] with random direction
        
        if self.random_start:
            player_pos = random_positions(k1, grid, n=1, exclude=walls_pos)
        else:
            player_pos = jnp.asarray([1, 1])
        
        direction = random_directions(k2, n=1)
        player = Player.create(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )

        entities = {
            Entities.PLAYER: player[None],
            Entities.WALL: walls,
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )

        from navix.environments.environment import Timestep

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(
        jnp.all(state.entities["player"].position == jnp.array([14, 14], dtype=jnp.int32)),
        dtype=jnp.float32,
    )


def symbolic(state: State) -> Array:
    H, W = state.grid.shape
    obs = jnp.zeros((H, W, 3), dtype=jnp.uint8)
    wall_symbol = jnp.array([EntityIds.WALL, 5, 0], dtype=jnp.uint8)
    floor_symbol = jnp.array([EntityIds.FLOOR, 0, 0], dtype=jnp.uint8)
    obs = jnp.where(state.grid[..., None] == -1, wall_symbol, floor_symbol)

    # Place entities
    for entity_class in state.entities:
        entity = state.entities[entity_class]
        tag = entity.tag
        if isinstance(entity, HasColour):
            colour = entity.colour
        else:
            colour = jnp.zeros(entity.shape)
        if isinstance(entity, Openable):
            entity_state = entity.open + (entity.requires != jnp.zeros(entity.shape))
        elif isinstance(entity, Directional):
            entity_state = entity.direction
        else:
            entity_state = jnp.zeros(entity.shape)
        entity_symbol = jnp.stack([tag, colour, entity_state], axis=-1, dtype=jnp.uint8)
        obs = obs.at[tuple(entity.position.T)].set(entity_symbol)

    return jnp.expand_dims(obs[:, :, 0] * 0.1 + obs[:, :, 2] * 0.01, axis=-1)


register_env(
    "gcrl_four_rooms",
    lambda *args, **kwargs: gcrl_FourRooms.create(
        height=16,
        width=16,
        random_start=kwargs.pop("random_start", True),
        observation_fn=kwargs.pop("observation_fn", symbolic),
        observation_space=kwargs.pop(
            "observation_space", Discrete.create(n_elements=9, shape=(16, 16, 1), dtype=jnp.float32)
        ),
        reward_fn=kwargs.pop("reward_fn", on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", free),  # Never terminate during training
        action_set=nx.actions.DEFAULT_ACTION_SET[:3],
        *args,
        **kwargs,
    ),
)
