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

# Import to register environments with navix
import mava.wrappers.navix_envs.nx_door_key
import mava.wrappers.navix_envs.nx_four_rooms
import mava.wrappers.navix_envs.nx_empty

from mava.wrappers.navix_envs.nx_door_key import gcrl_DoorKey, door_key_goal_coords
from mava.wrappers.navix_envs.nx_four_rooms import gcrl_FourRooms, four_rooms_goal_coords
from mava.wrappers.navix_envs.nx_empty import gcrl_empty, empty_goal_coords

__all__ = [
    "gcrl_DoorKey",
    "door_key_goal_coords",
    "gcrl_FourRooms",
    "four_rooms_goal_coords",
    "gcrl_empty",
    "empty_goal_coords",
]
