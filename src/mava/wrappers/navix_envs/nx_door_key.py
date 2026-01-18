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
from navix.components import EMPTY_POCKET_ID
from navix.rendering.cache import RenderingCache
from navix.rendering.registry import PALETTE
from navix.entities import Player, Wall, EntityIds, Door, Key
from navix.grid import random_positions, random_directions, room
from navix.spaces import Discrete
from navix.grid import mask_by_coordinates
from navix.environments.environment import Timestep

door_key_88_goal_coords = jnp.array([6, 6], dtype=jnp.int32)
door_key_goal_coords = jnp.array([14, 14], dtype=jnp.int32)


class gcrl_DoorKey(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None):
        assert self.height > 3, f"Room height must be greater than 3, got {self.height} instead"
        assert self.width > 4, f"Room width must be greater than 5, got {self.width} instead"

        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        grid = room(height=self.height, width=self.width)

        # Door position - divides room into left and right sections
        # For 16x16: door at column 8, for 8x8: door at column 4
        door_col = self.width // 2
        
        # if self.random_start:
        #     # Random door row (not at edges)
        #     door_row = jax.random.randint(k4, (), 2, self.height - 2)
        # else:
        #     # Fixed door position for easier learning
        door_row = self.height // 2
        
        door_pos = jnp.asarray((door_row, door_col))
        doors = Door.create(
            position=door_pos,
            requires=jnp.asarray(3),
            open=jnp.asarray(False),
            colour=PALETTE.YELLOW,
        )

        # Wall positions - vertical wall with door opening
        wall_rows = jnp.arange(1, self.height - 1)
        wall_cols = jnp.asarray([door_col] * (self.height - 2))
        wall_pos = jnp.stack((wall_rows, wall_cols), axis=1)
        wall_pos = jnp.delete(wall_pos, door_row - 1, axis=0, assume_unique_indices=True)
        walls = Wall.create(position=wall_pos)

        # Get room masks for spawning
        first_room_mask = mask_by_coordinates(grid, (jnp.asarray(self.height), door_col), jnp.less)
        first_room = jnp.where(first_room_mask, grid, -1)
        second_room_mask = mask_by_coordinates(grid, (jnp.asarray(0), door_col), jnp.greater)
        second_room = jnp.where(second_room_mask, grid, -1)

        if self.random_start:
            # Random player position in left room (before door)
            # random_positions with n=1 returns shape (2,) = [row, col]
            player_pos = random_positions(k1, first_room, n=1)
            # Random key position in left room (different from player)
            # Exclude player position by setting it to wall temporarily
            first_room_no_player = first_room.at[player_pos[0], player_pos[1]].set(-1)
            key_pos = random_positions(k3, first_room_no_player, n=1)
        else:
            # Fixed positions for easier learning
            player_pos = jnp.asarray([1, 1])
            key_pos = jnp.asarray([4, 1])
        
        # Player direction is always random
        player_dir = random_directions(k2)

        # Spawn player
        player = Player.create(position=player_pos, direction=player_dir, pocket=EMPTY_POCKET_ID)

        # Spawn key
        keys = Key.create(position=key_pos, id=jnp.asarray(3), colour=PALETTE.YELLOW)

        # Remove the wall at door position
        grid = grid.at[tuple(door_pos)].set(0)

        entities = {
            "player": player[None],
            "key": keys[None],
            "door": doors[None],
            "wall": walls,
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )


        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


def on_goal_reached_88(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(
        jnp.all(state.entities["player"].position == jnp.array([6, 6], dtype=jnp.int32)),
        dtype=jnp.float32,
    )


def on_goal_reached(prev_state: State, action: Array, state: State) -> Array:
    return jnp.asarray(
        jnp.all(state.entities["player"].position == jnp.array([14, 14], dtype=jnp.int32)),
        dtype=jnp.float32,
    )


def free(prev_state: State, action: Array, state: State) -> Array:
    return jnp.zeros_like(nx.events.on_ball_hit(state), dtype=jnp.bool_)


def terminate_on_goal(prev_state: State, action: Array, state: State) -> Array:
    """Terminate when agent reaches the goal [14, 14]."""
    at_goal = jnp.all(state.entities["player"].position == jnp.array([14, 14], dtype=jnp.int32))
    return jnp.asarray(at_goal, dtype=jnp.bool_)


def terminate_on_goal_88(prev_state: State, action: Array, state: State) -> Array:
    """Terminate when agent reaches the goal [6, 6] (for 8x8 grid)."""
    at_goal = jnp.all(state.entities["player"].position == jnp.array([6, 6], dtype=jnp.int32))
    return jnp.asarray(at_goal, dtype=jnp.bool_)


def symbolic(state: State) -> Array:
    from navix.components import Directional, HasColour, Openable

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
            entity_state = 1 - entity.open
        elif isinstance(entity, Directional):
            entity_state = entity.direction
        else:
            entity_state = jnp.zeros(entity.shape)
        entity_symbol = jnp.stack([tag, colour, entity_state], axis=-1, dtype=jnp.uint8)
        obs = obs.at[tuple(entity.position.T)].set(entity_symbol)

    return jnp.expand_dims(obs[:, :, 0] * 0.1 + obs[:, :, 2] * 0.01, axis=-1)


register_env(
    "gcrl_door_key-8x8",
    lambda *args, **kwargs: gcrl_DoorKey.create(
        height=8,
        width=8,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", symbolic),
        observation_space=kwargs.pop(
            "observation_space", Discrete.create(n_elements=9, shape=(8, 8, 1), dtype=jnp.float32)
        ),
        reward_fn=kwargs.pop("reward_fn", on_goal_reached_88),
        termination_fn=kwargs.pop("termination_fn", free),  # Never terminate during training
        action_set=nx.actions.DEFAULT_ACTION_SET[:6],
        *args,
        **kwargs,
    ),
)

register_env(
    "gcrl_door_key",
    lambda *args, **kwargs: gcrl_DoorKey.create(
        height=16,
        width=16,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", symbolic),
        observation_space=kwargs.pop(
            "observation_space", Discrete.create(n_elements=9, shape=(16, 16, 1), dtype=jnp.float32)
        ),
        reward_fn=kwargs.pop("reward_fn", on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", free),  # Never terminate during training
        action_set=nx.actions.DEFAULT_ACTION_SET[:6],
        *args,
        **kwargs,
    ),
)
