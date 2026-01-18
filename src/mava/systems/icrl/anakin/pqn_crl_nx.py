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

import copy
import os
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import chex
import numpy as np
import flax
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax

import mava.wrappers.navix_envs.nx_door_key
import mava.wrappers.navix_envs.nx_four_rooms
import mava.wrappers.navix_envs.nx_empty

import navix as nx
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from hydra.utils import instantiate
from jax import tree
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

# Mava imports
from mava.systems.icrl.types import Transition, ICRLParams, OptStates
from mava.types import ExperimentOutput
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.navix import NavixEnv
from mava.networks.icrl import sa_ConvEncoder_ActionInput, small_G_encoder

import flax.linen as nn


def flatten_crl_fn(buffer_config, transition, sample_key):
    """HER-style goal relabeling for Navix environments with grid observations."""
    gamma = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # Ensure goals from same episode
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
    )
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

    # Sample future goal indices and get goals from state_extras
    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    # Goals are stored in state_extras as (seq_len, 2) coordinates
    future_goals = jnp.take(transition.extras["state_extras"]["goal"], goal_index[:-1], axis=0)

    # Observations are grid format (seq_len, 1, H, W, 1) - squeeze agent dim
    obs = jnp.squeeze(transition.observation[:-1], axis=1)

    extras = {
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "seed": jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
        },
        "goal": future_goals,
    }

    return transition._replace(
        observation=obs,  # Grid observations (seq_len-1, H, W, 1)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


class NavixPQNLearnerState(NamedTuple):
    """Learner state for PQN-CRL with Navix environments."""

    params: ICRLParams
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: Any  # Navix/Brax-like State
    t: chex.Array
    target_params: ICRLParams = None


def soft_update(target_params, online_params, tau: float):
    """Soft update target network: target = (1 - tau) * target + tau * online"""
    return jax.tree_util.tree_map(
        lambda t, o: (1.0 - tau) * t + tau * o,
        target_params,
        online_params,
    )


def get_learner_fn(
    env,
    sa_encoder: nn.Module,
    g_encoder: nn.Module,
    critic_update_fn,
    config: DictConfig,
    env_goal_coords: jnp.ndarray,
) -> Any:
    """Get the learner function for PQN-CRL with Navix infrastructure."""

    # Extract config values
    num_envs = config.arch.num_envs
    rollout_length = config.system.rollout_length
    batch_size = config.system.batch_size
    gamma = config.system.gamma
    temperature = config.system.temperature
    epsilon = config.system.epsilon  # Epsilon-greedy exploration
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    use_target_network = config.system.use_target_network
    target_tau = config.system.target_tau

    # Observation dimensions
    obs_size = config.system.obs_size
    action_size = config.system.action_size

    # Buffer config - just gamma for HER discount
    buffer_config = gamma

    def compute_q_values(critic_params, obs, goal):

        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        batch_size_local = obs.shape[0]

        # Encode goal once
        g_repr = g_encoder.apply(g_encoder_params, goal)

        # Compute Q-value for each action using vmap (like Brax)
        def compute_q_for_action(action_idx):
            # Create one-hot encoding for this action
            a_onehot = jax.nn.one_hot(jnp.full(batch_size_local, action_idx), action_size)
            # Apply encoder with (obs, action_onehot)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, a_onehot)
            # Compute Q-value as negative L2 distance
            q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            return q

        # Vmap over all actions
        q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_size))
        q_values = q_values.T  # (action_size, batch) -> (batch, action_size)

        return q_values

    def get_action_params(learner_state: NavixPQNLearnerState):
        """Get params to use for action selection."""
        if use_target_network:
            return {
                "sa_encoder": learner_state.target_params.sa_encoder,
                "g_encoder": learner_state.target_params.goal_encoder,
            }
        else:
            return {
                "sa_encoder": learner_state.params.sa_encoder,
                "g_encoder": learner_state.params.goal_encoder,
            }

    # Pre-compute the fixed goal representation (broadcast to all envs)
    # This is the TARGET goal the agent is trying to reach
    fixed_goal = jnp.broadcast_to(env_goal_coords, (num_envs, 2))

    def actor_step(action_params, env_state, key, extra_fields):

        # obs has shape (num_envs, 1, H, W, 1) - squeeze agent dim
        obs = jnp.squeeze(env_state.obs, axis=1)  # Shape: (num_envs, H, W, 1)
        batch_size_local = obs.shape[0]

        # Compute Q-values using grid observations directly
        logits = compute_q_values(action_params, obs, fixed_goal)

        # All actions are available in Navix - create directly with correct shape
        avail_actions = jnp.ones((batch_size_local, action_size))
        masked_logits = logits - ((1 - avail_actions) * 1e10)

        # Sample actions with temperature (Boltzmann exploration)
        key, action_key, epsilon_key = jax.random.split(key, 3)
        policy_actions = jax.random.categorical(action_key, masked_logits / temperature, axis=-1)

        # Epsilon-greedy: with probability epsilon, take random action
        random_actions = jax.random.randint(epsilon_key, (batch_size_local,), 0, action_size)
        use_random = jax.random.uniform(key, (batch_size_local,)) < epsilon
        actions = jnp.where(use_random, random_actions, policy_actions)

        nstate = env.step(env_state, policy_actions)

        # Store state extras and goal from NEXT state player position
        state_extras = {x: nstate.info[x] for x in extra_fields}
        # Extract player position from next state: shape (num_envs, 1, 2) -> squeeze to (num_envs, 2)
        state_extras['goal'] = jnp.squeeze(nstate.pipeline_state.state.entities['player'].position, axis=1)

        return nstate, Transition(
            observation=env_state.obs,  # Grid observation (num_envs, 1, H, W, 1)
            action=policy_actions,
            avail_actions=avail_actions,
            reward=nstate.reward,  # Shape: (num_envs,)
            discount=1.0 - nstate.done.astype(jnp.float32),  # Shape: (num_envs,)
            extras={"state_extras": state_extras},
        )

    def collect_experience(action_params, env_state, key):

        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)

            env_state, transition = actor_step(
                action_params,
                env_state,
                current_key,
                extra_fields=("truncation", "seed"),
            )
            return (env_state, next_key), transition

        (env_state, _), transitions = jax.lax.scan(f, (env_state, key), (), length=rollout_length)
        return env_state, transitions

    def update_critic_pqn(transitions, learner_state, key):

        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]

            # Observations are already in grid format (batch, H, W, 1)
            obs = transitions.observation
            action = transitions.action
            goal = transitions.extras["goal"]  # Use the relabeled goal from HER

            batch_size_local = obs.shape[0]

            # Encode goal
            g_repr = g_encoder.apply(g_encoder_params, goal)

            # Get representation for the taken action (for contrastive loss)
            action_onehot = jax.nn.one_hot(action, action_size)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action_onehot)

            # InfoNCE loss
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            critic_loss_val = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # Logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss_val += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            # Compute policy entropy by computing Q-values for all actions (vmap)
            def compute_q_for_action(action_idx):
                a_onehot = jax.nn.one_hot(jnp.full(batch_size_local, action_idx), action_size)
                sa_repr_a = sa_encoder.apply(sa_encoder_params, obs, a_onehot)
                q = -jnp.sqrt(jnp.sum((sa_repr_a - g_repr) ** 2, axis=-1))
                return q

            q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_size))
            q_values = q_values.T  # (action_size, batch) -> (batch, action_size)

            policy = jax.nn.softmax(q_values / temperature, axis=-1)
            log_policy = jnp.log(policy + 1e-8)
            policy_entropy = -jnp.sum(policy * log_policy, axis=-1)
            mean_entropy = jnp.mean(policy_entropy)
            max_entropy = jnp.log(action_size)
            entropy_ratio = mean_entropy / max_entropy

            return critic_loss_val, (logsumexp, correct, logits_pos, logits_neg, mean_entropy, entropy_ratio)

        critic_params = {
            "sa_encoder": learner_state.params.sa_encoder,
            "g_encoder": learner_state.params.goal_encoder,
        }

        (loss, (logsumexp, correct, logits_pos, logits_neg, mean_entropy, entropy_ratio)), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            critic_params, transitions, key
        )

        # Average gradients across devices
        grad = lax.pmean(grad, axis_name="device")

        new_critic_state = learner_state.opt_states.critic
        updates, new_critic_state = critic_update_fn(grad, new_critic_state)

        new_sa_encoder = optax.apply_updates(learner_state.params.sa_encoder, updates["sa_encoder"])
        new_g_encoder = optax.apply_updates(learner_state.params.goal_encoder, updates["g_encoder"])

        new_params = ICRLParams(
            sa_encoder=new_sa_encoder,
            goal_encoder=new_g_encoder,
        )
        new_opt_states = OptStates(critic=new_critic_state)

        # Update target network if using
        if use_target_network:
            new_target_params = soft_update(
                {
                    "sa_encoder": learner_state.target_params.sa_encoder,
                    "g_encoder": learner_state.target_params.goal_encoder,
                },
                {"sa_encoder": new_sa_encoder, "g_encoder": new_g_encoder},
                target_tau,
            )
            new_target_params = ICRLParams(
                sa_encoder=new_target_params["sa_encoder"],
                goal_encoder=new_target_params["g_encoder"],
            )
        else:
            new_target_params = learner_state.target_params

        learner_state = NavixPQNLearnerState(
            params=new_params,
            opt_states=new_opt_states,
            key=learner_state.key,
            env_state=learner_state.env_state,
            t=learner_state.t,
            target_params=new_target_params,
        )

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
            "policy_entropy": mean_entropy,
            "entropy_ratio": entropy_ratio,
        }

        return learner_state, metrics

    def sgd_step(carry, transitions):
        """Single SGD step on a mini-batch."""
        learner_state, key = carry
        key, critic_key = jax.random.split(key, 2)
        learner_state, metrics = update_critic_pqn(transitions, learner_state, critic_key)
        return (learner_state, key), metrics

    def _update_step(learner_state: NavixPQNLearnerState, _: Any) -> Tuple[NavixPQNLearnerState, Tuple]:
        key = learner_state.key
        experience_key, train_key, sampling_key, permute_key, new_key = jax.random.split(key, 5)

        # Collect fresh experience
        env_state, transitions = collect_experience(
            get_action_params(learner_state),
            learner_state.env_state,
            experience_key,
        )

        # Transpose transitions
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1),
            transitions,
        )

        # Apply hindsight relabeling
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_crl_fn, in_axes=(None, 0, 0))(
            buffer_config, transitions, batch_keys
        )

        # Flatten the batch dimension
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        # Shuffle
        permutation = jax.random.permutation(permute_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        # Truncate to make evenly divisible by batch_size
        num_samples = (len(transitions.observation) // batch_size) * batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_samples], transitions)

        # Reshape into mini-batches for training
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )

        # Update learner state with new env_state
        learner_state = NavixPQNLearnerState(
            params=learner_state.params,
            opt_states=learner_state.opt_states,
            key=new_key,
            env_state=env_state,
            t=learner_state.t + num_envs * rollout_length,
            target_params=learner_state.target_params,
        )

        # Train on all mini-batches
        (learner_state, _), metrics = jax.lax.scan(sgd_step, (learner_state, train_key), transitions)

        return learner_state, ({}, metrics)

    def learner_fn(learner_state: NavixPQNLearnerState) -> ExperimentOutput:
        """Learner function - performs multiple update steps."""
        learner_state, (episode_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None, config.system.num_updates_per_eval
        )

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_metrics,
            train_metrics=train_metrics,
        )

    def eval_actor_step(critic_params, env_state, g_repr, key, temperature):
        """Evaluation step - use fixed goal representation with vmap over actions."""
        # obs has shape (num_eval_envs, 1, H, W, 1) - squeeze agent dim
        obs = jnp.squeeze(env_state.obs, axis=1)  # Shape: (num_eval_envs, H, W, 1)
        batch_size_local = obs.shape[0]

        # Compute Q-values using vmap over actions (like Brax)
        sa_encoder_params = critic_params["sa_encoder"]

        def compute_q_for_action(action_idx):
            a_onehot = jax.nn.one_hot(jnp.full(batch_size_local, action_idx), action_size)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, a_onehot)
            q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            return q

        q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_size))
        q_values = q_values.T  # (action_size, batch) -> (batch, action_size)

        # Greedy action selection
        # actions = jnp.argmax(q_values, axis=-1)
        actions =  jax.random.categorical(key, q_values / temperature, axis=-1)
        nstate = env.step(env_state, actions)

        return nstate

    return learner_fn, eval_actor_step


def learner_setup(config: DictConfig, env_goal_coords: jnp.ndarray) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states."""
    n_devices = len(jax.devices())

    # PRNG keys
    key = jax.random.PRNGKey(config.system.seed)
    key, env_key, eval_env_key, sa_key, g_key = jax.random.split(key, 5)

    # Environment setup
    env = NavixEnv(env_name=config.env.env_name)

    # Store dimensions in config
    config.system.action_size = env.action_size
    # Grid shape for ConvEncoder
    grid_shape = tuple(config.env.grid_shape)
    config.system.grid_shape = grid_shape
    # obs_size kept for compatibility - observations are grids (H, W, 1)
    config.system.obs_size = int(jnp.prod(jnp.array(grid_shape)))

    # Initialize environment states
    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, n_devices * num_envs)
    env_keys = env_keys.reshape(n_devices, num_envs, -1)

    # Create env states for each device - env.reset is already vmapped
    def init_envs(keys):
        return env.reset(keys)

    env_states = jax.vmap(init_envs)(env_keys)

    # Create networks - use new encoder that takes (obs, action_onehot) as input
    sa_encoder = sa_ConvEncoder_ActionInput(rep_size=config.system.rep_size)
    g_encoder = small_G_encoder(rep_size=config.system.rep_size)

    # Initialize network parameters
    dummy_obs = jnp.ones([1, *grid_shape, 1])
    dummy_action = jnp.ones([1, config.system.action_size])
    sa_encoder_params = sa_encoder.init(sa_key, dummy_obs, dummy_action)

    dummy_goal = jnp.ones([1, 2])
    g_encoder_params = g_encoder.init(g_key, dummy_goal)

    # Pack parameters
    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=g_encoder_params,
    )
    target_params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=g_encoder_params,
    )

    # Optimizer
    critic_opt = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(learning_rate=config.system.q_lr),
    )
    critic_opt_state = critic_opt.init({"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params})
    opt_states = OptStates(critic=critic_opt_state)

    # Get learner function
    learn, eval_actor_step = get_learner_fn(env, sa_encoder, g_encoder, critic_opt.update, config, env_goal_coords)

    # Pmap the learner
    learn = jax.pmap(learn, axis_name="device")

    # Replicate params and opt_states across devices
    params = flax.jax_utils.replicate(params, devices=jax.devices())
    target_params = flax.jax_utils.replicate(target_params, devices=jax.devices())
    opt_states = flax.jax_utils.replicate(opt_states, devices=jax.devices())

    # Create step keys for each device
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices)

    # Initialize timestep counter
    t0 = jnp.zeros((n_devices,), dtype=jnp.int32)

    # Create initial learner state
    init_learner_state = NavixPQNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        t=t0,
        target_params=target_params,
    )

    return learn, sa_encoder, g_encoder, init_learner_state, eval_actor_step, env


def render_best_policy(
    config: DictConfig,
    best_policy_path: str,
    save_dir: str,
    env_goal_coords: jnp.ndarray,
    num_episodes: int = 3,
    max_steps: int = 100,
    temperature: float = 0.1,
) -> Optional[str]:

    import os
    import pickle
    
    try:
        import imageio
        import navix as nx
        from navix.rendering.cache import render_background, TILE_SIZE
        from navix.rendering.registry import SPRITES_REGISTRY
    except ImportError as e:
        print(f"Warning: Could not import rendering modules: {e}. Skipping GIF generation.")
        return None
    
    # Load parameters
    if not os.path.exists(best_policy_path):
        print(f"Warning: Best policy not found at {best_policy_path}. Skipping GIF generation.")
        return None
        
    with open(best_policy_path, 'rb') as f:
        params = pickle.load(f)
    
    sa_params = params['sa_encoder']
    g_params = params['goal_encoder']
    
    # Setup environment
    env_name = config.env.env_name
    env = nx.make(env_name)
    action_size = len(env.action_set)
    rep_size = config.system.rep_size
    
    # Setup networks
    sa_encoder = sa_ConvEncoder_ActionInput(rep_size=rep_size)
    g_encoder = small_G_encoder(rep_size=rep_size)
    
    # Encode goal once
    goal_input = env_goal_coords.astype(jnp.float32)[None, :]
    g_repr = g_encoder.apply(g_params, goal_input)[0]
    
    def render_state(state, goal_coords=None):
        """Render a Navix state to an RGB image."""
        grid = state.grid
        H, W = grid.shape
        
        # Start with background (floor and walls)
        image = np.array(render_background(grid))
        
        entities = state.entities
        
        # Render player
        if 'player' in entities:
            player = entities['player']
            pos = np.array(player.position).flatten()
            if len(pos) >= 2:
                y, x = int(pos[0]), int(pos[1])
                if 0 <= y < H and 0 <= x < W:
                    direction = int(np.array(player.direction).flatten()[0]) % 4
                    player_sprite = np.array(SPRITES_REGISTRY['player'][direction])
                    y_start, x_start = y * TILE_SIZE, x * TILE_SIZE
                    y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
                    img_h, img_w = image.shape[:2]
                    if y_end <= img_h and x_end <= img_w:
                        mask = np.any(player_sprite > 0, axis=-1, keepdims=True)
                        image[y_start:y_end, x_start:x_end] = np.where(
                            mask, player_sprite, image[y_start:y_end, x_start:x_end]
                        )
        
        # Render door if present
        if 'door' in entities:
            door = entities['door']
            pos = np.array(door.position).flatten()
            if len(pos) >= 2:
                y, x = int(pos[0]), int(pos[1])
                if 0 <= y < H and 0 <= x < W:
                    colour = int(np.array(door.colour).flatten()[0]) if hasattr(door, 'colour') else 0
                    is_open = int(np.array(door.open).flatten()[0]) if hasattr(door, 'open') else 0
                    door_sprite = np.array(SPRITES_REGISTRY['door'][colour, is_open])
                    y_start, x_start = y * TILE_SIZE, x * TILE_SIZE
                    y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
                    img_h, img_w = image.shape[:2]
                    if y_end <= img_h and x_end <= img_w:
                        mask = np.any(door_sprite > 0, axis=-1, keepdims=True)
                        image[y_start:y_end, x_start:x_end] = np.where(
                            mask, door_sprite, image[y_start:y_end, x_start:x_end]
                        )
        
        # Render key if present
        if 'key' in entities:
            key_entity = entities['key']
            pos = np.array(key_entity.position).flatten()
            if len(pos) >= 2:
                y, x = int(pos[0]), int(pos[1])
                if 0 <= y < H and 0 <= x < W:
                    colour = int(np.array(key_entity.colour).flatten()[0]) if hasattr(key_entity, 'colour') else 0
                    key_sprite = np.array(SPRITES_REGISTRY['key'][colour])
                    y_start, x_start = y * TILE_SIZE, x * TILE_SIZE
                    y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
                    img_h, img_w = image.shape[:2]
                    if y_end <= img_h and x_end <= img_w:
                        mask = np.any(key_sprite > 0, axis=-1, keepdims=True)
                        image[y_start:y_end, x_start:x_end] = np.where(
                            mask, key_sprite, image[y_start:y_end, x_start:x_end]
                        )
        
        # Highlight goal
        if goal_coords is not None:
            goal_y, goal_x = int(goal_coords[0]), int(goal_coords[1])
            y_start, x_start = goal_y * TILE_SIZE, goal_x * TILE_SIZE
            y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
            img_h, img_w = image.shape[:2]
            y_end, x_end = min(y_end, img_h), min(x_end, img_w)
            if y_end > y_start and x_end > x_start:
                # Green tint for goal
                image[y_start:y_end, x_start:x_end, 1] = np.minimum(
                    255, image[y_start:y_end, x_start:x_end, 1] + 50
                )
        
        return image.astype(np.uint8)
    
    # Collect frames
    all_frames = []
    key = jax.random.PRNGKey(42)
    
    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)
        timestep = env.reset(reset_key)
        frames = [render_state(timestep.state, env_goal_coords)]
        
        for step in range(max_steps):
            obs = timestep.observation[None, ...]  # Add batch dim
            
            # Compute Q-values
            def compute_q(action_idx):
                a_onehot = jax.nn.one_hot(jnp.array([action_idx]), action_size)
                sa_repr = sa_encoder.apply(sa_params, obs, a_onehot)[0]
                return -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2))
            
            q_values = jax.vmap(compute_q)(jnp.arange(action_size))
            
            # Sample action
            key, action_key = jax.random.split(key)
            action = jax.random.categorical(action_key, q_values / temperature)
            
            timestep = env.step(timestep, int(action))
            frames.append(render_state(timestep.state, env_goal_coords))
            
            # Check if goal reached
            player_pos = timestep.state.entities["player"].position
            if jnp.all(player_pos == env_goal_coords):
                break
        
        all_frames.extend(frames)
        # Add separator frames between episodes
        if ep < num_episodes - 1:
            all_frames.extend([all_frames[-1]] * 10)
    
    # Save GIF
    gif_path = os.path.join(save_dir, "policy_rollout.gif")
    imageio.mimsave(gif_path, all_frames, fps=10, loop=0)
    print(f"Saved policy rollout GIF: {gif_path}")
    
    return gif_path


def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "pqn_crl_nx"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Get environment-specific goal coordinates
    env_name = config.env.env_name
    if env_name == "gcrl_door_key":
        from mava.wrappers.navix_envs.nx_door_key import door_key_goal_coords as env_goal_coords
    elif env_name == "gcrl_four_rooms":
        from mava.wrappers.navix_envs.nx_four_rooms import four_rooms_goal_coords as env_goal_coords
    elif env_name == "gcrl_empty":
        from mava.wrappers.navix_envs.nx_empty import empty_goal_coords as env_goal_coords
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Setup learner and environment
    learn, sa_encoder, g_encoder, learner_state, eval_actor_step, env = learner_setup(config, env_goal_coords)

    jax.block_until_ready(learner_state)

    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates must be greater than number of evaluations."

    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices * config.system.num_updates_per_eval * config.system.rollout_length * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )

    # Setup evaluation
    num_eval_envs = config.arch.num_eval_envs
    eval_episode_length = config.env.time_limit
    eval_key = jax.random.PRNGKey(config.system.seed + 1000)

    # Pre-encode the fixed goal for evaluation
    g_encoder_params_init = jax.tree_util.tree_map(lambda x: x[0], learner_state.params.goal_encoder)
    
    @jax.jit
    def run_evaluation(critic_params, g_encoder_params, key):
        """Run evaluation episodes with fixed goal."""
        # Encode the fixed goal (cast to float32 and add batch dimension)
        goal_input = env_goal_coords.astype(jnp.float32)[None, :]  # Shape: (1, 2)
        g_repr = g_encoder.apply(g_encoder_params, goal_input)  # Shape: (1, rep_size)
        g_repr = jnp.broadcast_to(g_repr, (num_eval_envs, g_repr.shape[-1]))  # Broadcast to batch
        temperature = config.system.get("temperature", 0.1)
        # Reset evaluation environments
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_eval_envs)
        eval_first_state = env.reset(reset_keys)
        
        # Run single episode per env - stop stepping once goal is reached
        def eval_step_fn(carry, step_idx):
            env_state, episode_done, episode_return, episode_length, final_distance, step_key = carry
            step_key, next_key = jax.random.split(step_key)
            
            # Only step envs that haven't finished
            nstate = eval_actor_step(critic_params, env_state, g_repr, step_key, temperature)
            
            # Check if goal reached (reward > 0)
            goal_reached = nstate.reward > 0
            
            # Update episode done status
            new_episode_done = jnp.logical_or(episode_done, goal_reached)
            
            # Update return only for envs that aren't done yet
            new_episode_return = jnp.where(episode_done, episode_return, episode_return + nstate.reward)
            
            # Record episode length when first reaching goal
            new_episode_length = jnp.where(
                jnp.logical_and(goal_reached, ~episode_done),
                step_idx + 1,  # Current step (1-indexed)
                episode_length
            )
            
            # Record final distance when episode ends
            current_pos = nstate.pipeline_state.state.entities['player'].position[:, 0, :]
            current_distance = jnp.sum(jnp.abs(current_pos - env_goal_coords), axis=-1)
            new_final_distance = jnp.where(
                jnp.logical_and(goal_reached, ~episode_done),
                current_distance,
                final_distance
            )
            
            return (nstate, new_episode_done, new_episode_return, new_episode_length, new_final_distance, next_key), None
        
        key, unroll_key = jax.random.split(key)
        initial_episode_done = jnp.zeros(num_eval_envs, dtype=jnp.bool_)
        initial_episode_return = jnp.zeros(num_eval_envs)
        initial_episode_length = jnp.full(num_eval_envs, eval_episode_length)  # Default to max if never succeed
        initial_final_distance = jnp.full(num_eval_envs, 100.0)  # Large default distance
        
        (eval_final_state, episode_done, episode_return, episode_length, final_distance, _), _ = jax.lax.scan(
            eval_step_fn, 
            (eval_first_state, initial_episode_done, initial_episode_return, initial_episode_length, initial_final_distance, unroll_key), 
            jnp.arange(eval_episode_length),  # Pass step index
            length=eval_episode_length
        )
        
        # For envs that never succeeded, compute final distance at end
        final_pos = eval_final_state.pipeline_state.state.entities['player'].position[:, 0, :]
        end_distance = jnp.sum(jnp.abs(final_pos - env_goal_coords), axis=-1)
        final_distance = jnp.where(episode_done, final_distance, end_distance)
        
        # Compute metrics
        first_pos = eval_first_state.pipeline_state.state.entities['player'].position[:, 0, :]
        distance_from_start = jnp.sum(jnp.abs(final_pos - first_pos), axis=-1)
        
        # Success = episode completed (reached goal)
        success = episode_done.astype(jnp.float32)
        
        return {
            "episode_return": episode_return,
            "distance_from_start": distance_from_start,
            "distance_from_goal": final_distance,
            "win_rate": success * 100,
            "episode_length": episode_length,
        }, key

    max_success_rate = -jnp.inf
    best_params = None
    
    # Create directory for saving best policy
    save_dir = f"results/{config.env.env_name}"
    os.makedirs(save_dir, exist_ok=True)

    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        # Process training metrics
        train_metrics = learner_output.train_metrics
        train_metrics = jax.tree_util.tree_map(jnp.mean, train_metrics)

        # Calculate steps per second
        sps = steps_per_rollout / elapsed_time

        # Log training metrics
        train_metrics["steps_per_second"] = sps

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Run evaluation
        eval_start_time = time.time()
        
        # Get unreplicated params for evaluation (just take first device's params)
        eval_sa_params = jax.tree_util.tree_map(lambda x: x[0], learner_output.learner_state.params.sa_encoder)
        eval_g_params = jax.tree_util.tree_map(lambda x: x[0], learner_output.learner_state.params.goal_encoder)
        eval_critic_params = {"sa_encoder": eval_sa_params, "g_encoder": eval_g_params}
        
        eval_results, eval_key = run_evaluation(eval_critic_params, eval_g_params, eval_key)
        eval_elapsed = time.time() - eval_start_time
        
        # Aggregate evaluation metrics
        eval_metrics = {
            "episode_return": float(jnp.mean(eval_results["episode_return"])),
            "distance_from_start": float(jnp.mean(eval_results["distance_from_start"])),
            "distance_from_goal": float(jnp.mean(eval_results["distance_from_goal"])),
            "win_rate": float(jnp.mean(eval_results["win_rate"])),
            "success_any": float(jnp.mean(eval_results["episode_return"] > 0)),
            "avg_episode_length": float(jnp.mean(eval_results["episode_length"])),
            "eval_time": eval_elapsed,
            "eval_sps": (num_eval_envs * eval_episode_length) / eval_elapsed,
        }
        
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            episode_return = eval_metrics["episode_return"]
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=jax.tree_util.tree_map(lambda x: x[0], learner_output.learner_state),
                episode_return=episode_return,
            )

        # Track and save best policy based on success rate
        current_success_rate = eval_metrics["win_rate"]
        if current_success_rate > max_success_rate:
            max_success_rate = current_success_rate
            best_params = {
                "sa_encoder": jax.tree_util.tree_map(lambda x: np.array(x), eval_sa_params),
                "goal_encoder": jax.tree_util.tree_map(lambda x: np.array(x), eval_g_params),
            }
            # Save best policy
            best_policy_path = f"{save_dir}/best_policy.pkl"
            with open(best_policy_path, 'wb') as f:
                pickle.dump(best_params, f)
            print(f"New best policy saved! Win rate: {current_success_rate:.2f}% -> {best_policy_path}")

        learner_state = learner_output.learner_state

    # Return final performance metric 
    eval_performance = float(eval_metrics.get("win_rate", 0.0))

    # Print summary
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best win rate: {max_success_rate:.2f}%")
    print(f"Best policy saved to: {save_dir}/best_policy.pkl")
    print(f"{'='*50}\n")

    # Render best policy and upload GIF to Neptune
    gif_path = render_best_policy(
        config=config,
        best_policy_path=f"{save_dir}/best_policy.pkl",
        save_dir=save_dir,
        env_goal_coords=env_goal_coords,
    )
    if gif_path:
        logger.upload_artifact("artifacts/policy_rollout", gif_path)
        print(f"Policy rollout GIF uploaded to Neptune: {gif_path}")

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="pqn_crl_nx.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    try:
        eval_performance = run_experiment(cfg)
        return eval_performance
    except Exception as e:
        try:
            override_info = cfg.hydra.job.override_dirname
        except Exception:
            override_info = "unknown"
        print(f"Error executing job with overrides: {override_info}")
        print(f"Exception: {type(e).__name__}: {e!s}")
        import traceback

        traceback.print_exc()
        return float("-inf")


if __name__ == "__main__":
    hydra_entry_point()
