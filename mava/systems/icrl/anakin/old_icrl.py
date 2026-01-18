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
import time
from typing import Any, Tuple
import flax
import flax.linen as nn
import chex
import distrax
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.networks import SAEncoder, GoalEncoder, ICRLActor
from mava.systems.icrl.types import ICRLParams, OptStates, LearnerState, Transition as ICRLTransition
from mava.systems.crl.evaluator import CrlEvaluator
from mava.types import ExperimentOutput
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue
from brax import envs


def get_learner_fn(
    env: Any,  # Brax env (not MarlEnv)
    buffer: TrajectoryUniformSamplingQueue,
    apply_fns: Tuple,
    update_fns: Tuple,
    config: DictConfig,
    get_avail_act: Any,  # Function to get available actions
    ) -> Any:
    """Get the learner function."""
    # Unpack apply and update functions
    sa_encoder_apply, goal_encoder_apply, actor_apply = apply_fns
    actor_update_fn, critic_update_fn, alpha_update_fn = update_fns

    # Multi-agent dimensions - use from config (set before wrapping)
    n_agents = config.system.num_agents
    action_dim = env.action_size
    # num_envs_agents will be calculated from state shape in single_step for robustness

    # Goal relabeling parameters (for loss functions)
    # obs_dim should be base + current_health = 128 (excludes ultimate_goal)
    # config.system.icrl.obs_dim is just the base (127)
    observation_size = config.system.icrl.obs_dim + config.system.icrl.goal_dim
    obs_dim = observation_size - 1
    goal_start_idx = observation_size - 2
    goal_end_idx = observation_size - 1

    # Target entropy for SAC
    target_entropy = -action_dim / 2
    
    # Logsumexp penalty coefficient
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff

    def _env_step(
        learner_state: LearnerState, _: Any
        ) -> Tuple[LearnerState, dict]:
        """Step the environment for rollout_length steps."""
        params, opt_states, buffer_state, key, env_state = learner_state

        def single_step(carry, _):
            """Single environment step (Brax interface)."""
            key, env_state, buffer_state = carry

            # RNG
            key, policy_key = jax.random.split(key)
            
            # Get obs from Brax State (like CRL line 420)
            # Infer num_envs from state shape (more robust to batching)
            num_envs_actual = env_state.obs.shape[0]  # First dim is num_envs
            num_envs_agents = env_state.obs.shape[0] * env_state.obs.shape[1]  # num_envs * num_agents
            obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])  # [N_env*N_agent, obs_dim]
            
            # Reshape for actor (needs [N_env, N_agent, obs_dim])
            obs_reshaped = obs.reshape(num_envs_actual, n_agents, -1)
            means, log_stds = actor_apply(params.actor, obs_reshaped)  # means: [N_env, N_agent, A]
            stds = jnp.exp(log_stds)

            # Get available actions (like CRL line 434)
            avail_actions = get_avail_act(env_state)  # [N_env, N_agent, action_dim]
            avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])  # [N_env*N_agent, action_dim]
            avail_actions = jnp.asarray(avail_actions, dtype=means.dtype)
            
            # Flatten means and apply masking (matching CRL line 435-436)
            means_flat = means.reshape(-1, means.shape[-1])  # [N_env*N_agent, action_dim]
            stds_flat = stds.reshape(-1, stds.shape[-1])  # [N_env*N_agent, action_dim]
            means_flat = means_flat - (1.0 - avail_actions) * 1e10  # -inf on illegal

            # GUMBEL-SOFTMAX TRICK: Treat discrete actions as continuous (matching CRL line 428-431)
            noise_keys = jax.random.split(policy_key, n_agents)
            # Reshape for per-agent noise generation: [N_agent, N_env, action_dim]
            means_for_noise = means_flat.reshape(num_envs_actual, n_agents, -1).transpose(1, 0, 2)
            noise = jax.vmap(lambda rng: jax.random.gumbel(
                rng, shape=(num_envs_actual,) + means_for_noise.shape[2:],
                dtype=means.dtype
            ), in_axes=0, out_axes=1)(noise_keys)  # [N_agent, N_env, action_dim]
            noise_flat = noise.transpose(1, 0, 2).reshape(-1, noise.shape[-1])  # [N_env*N_agent, action_dim]
            
            # Sample continuous actions using Gumbel noise + tanh (matching CRL line 438)
            action_continuous_flat = nn.tanh(means_flat + stds_flat * noise_flat)  # [N_env*N_agent, action_dim]
            
            # Reshape actions for env step (like CRL line 451)
            # Pass continuous actions - env will do argmax internally for discrete actions
            actions_ = jnp.reshape(action_continuous_flat, (-1, n_agents) + action_continuous_flat.shape[1:])

            # Step env (returns State, not tuple) - like CRL line 453
            nstate = env.step(env_state, actions_)

            # Extract from Brax State (like CRL lines 456, 462-463)
            flat_obs = obs  # Already flattened
            flat_next_obs = jnp.reshape(nstate.obs, (-1,) + nstate.obs.shape[2:])
            flat_action = action_continuous_flat  # Store continuous actions (already flattened)
            flat_reward = jnp.repeat(nstate.reward, n_agents)  # Replicate scalar reward
            flat_discount = jnp.repeat(1 - nstate.done, n_agents)  # Compute from done
            
            # Store avail_actions for actor loss masking
            flat_avail = avail_actions  # Already flattened

            # Extract extras from state.info (like CRL line 456)
            trunc = jnp.repeat(nstate.info["truncation"], n_agents)  # Replicate scalar
            seed = jnp.repeat(nstate.info["seed"], n_agents)  # Replicate scalar

            transition = ICRLTransition(
                observation=flat_obs,
                action=flat_action,  # keep consistent with your critic (one-hot vs indices)
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,  # Store availability mask
                extras={"state_extras": {"truncation": trunc, "seed": seed}},
            )

            return (key, nstate, buffer_state), transition


        # Collect rollout_length transitions
        (key, env_state, buffer_state), traj_batch = jax.lax.scan(
            single_step, (key, env_state, buffer_state), None, config.system.rollout_length
        )    
        # Add trajectory to buffer (time-major format)
        buffer_state = buffer.insert(buffer_state, traj_batch)
        
        # Get episode metrics from state (Brax doesn't have episode_metrics the same way)
        # For SMAX, success is tracked in state.metrics
        metrics = {}  # Extract metrics if needed from env_state.metrics
        
        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state)
        return learner_state, metrics

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network (collect rollout + train)."""
        
        # Collect experience
        learner_state, episode_metrics = _env_step(learner_state, None)
        
        # Sample batch from buffer
        params, opt_states, buffer_state, key, env_state = learner_state
        key, sample_key = jax.random.split(key)
        buffer_state, transitions = buffer.sample(buffer_state)
        
        # Apply hindsight relabeling (flatten_crl_fn)
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        transitions = jax.vmap(
            TrajectoryUniformSamplingQueue.flatten_crl_fn,
            in_axes=(None, 0, 0)
        )(
            (config.system.gamma, obs_dim, goal_start_idx, goal_end_idx),
            transitions,
            batch_keys
        )
        
        # Reshape with Fortran order (like original)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        
        # Randomly permute transitions
        perm_key, sample_key = jax.random.split(sample_key)
        permutation = jax.random.permutation(perm_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        
        # Truncate to make evenly divisible by batch_size
        num_samples = (len(transitions.observation) // config.system.batch_size) * config.system.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_samples], transitions)
        
        # Reshape into batches of batch_size (CRITICAL for proper gradient scaling!)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
            transitions,
        )

        def _update_minibatch(carry, batch_transitions):
            """Update networks on a single minibatch."""
            params, opt_states, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)
            
            # Extract obs and action for this batch
            obs = batch_transitions.observation  # [batch_size, obs_dim+goal_dim]
            action = batch_transitions.action    # [batch_size, action_dim]

            def _critic_loss_fn(critic_params, obs, action):
                """InfoNCE contrastive loss for critic."""
                # Split observation into state and goal (matching CRL exactly, line 572-577)
                # After relabeling: new_obs = concat([state, goal]) where state=[0:obs_dim], goal=[obs_dim:]
                state = obs[:, :obs_dim]   # [batch, 128] (base_obs + current_health)
                goal = obs[:, obs_dim:]    # [batch, 1] (goal from future_state, placed at obs_dim after relabeling)
                
                # Compute representations
                sa_repr = sa_encoder_apply(critic_params['sa_encoder'], state, action)
                g_repr = goal_encoder_apply(critic_params['goal_encoder'], goal)
                
                # InfoNCE: compute pairwise distances with epsilon for numerical stability
                # logits[i,j] = -distance(sa_repr[i], g_repr[j])
                logits = -jnp.sqrt(
                    jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1) + 1e-8
                )  # [batch, batch]
                
                # InfoNCE loss: maximize diagonal (positive pairs), minimize off-diagonal
                critic_loss = -jnp.mean(
                    jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1)
                )
                
                # Logsumexp regularization (from original ICRL)
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
                
                # Metrics
                logits_pos = jnp.diag(logits).mean()
                logits_neg = (logits.sum() - jnp.diag(logits).sum()) / (logits.size - logits.shape[0])
                categorical_accuracy = (logits.argmax(axis=1) == jnp.arange(logits.shape[0])).mean()
                
                loss_info = {
                    "critic_loss": critic_loss,
                    "logits_pos": logits_pos,
                    "logits_neg": logits_neg,
                    "categorical_accuracy": categorical_accuracy,
                }
                return critic_loss, loss_info

            def _actor_loss_fn(actor_params, critic_params, obs, alpha, avail_actions, future_state, key):
                """Actor loss with Gumbel-Softmax (matching CRL implementation exactly)."""
                # Extract state from observation and goal from future_state (matching CRL line 505-507)
                state = obs[:, :obs_dim]
                goal = future_state[:, goal_start_idx : goal_end_idx]  # Match CRL exactly: future_state[:, 127:128]
                observation = jnp.concatenate([state, goal], axis=1)
                
                means, log_stds = actor_apply(actor_params, observation)
                avail_mask = avail_actions.astype(means.dtype)

                if config.system.discrete_actions:
                    action_logits = means - ((1.0 - avail_mask) * 1e10)
                    categorical = distrax.Categorical(logits=action_logits)
                    action_idx = categorical.sample(seed=key)
                    log_prob = categorical.log_prob(action_idx)
                    action = jax.nn.one_hot(action_idx, action_logits.shape[-1])
                else:
                    # GUMBEL-SOFTMAX: Match CRL exactly (lines 516-527)
                    stds = jnp.exp(log_stds)
                    # Apply masking before Gumbel sampling (CRL line 518)
                    means = means - ((1.0 - avail_mask) * 1e10)
                    x_ts = means + stds * jax.random.gumbel(key, shape=means.shape, dtype=means.dtype)
                    action = nn.tanh(x_ts)

                    # Gumbel log PDF (CRL line 522-523)
                    x_std = (x_ts - means) / stds
                    log_prob = -(x_std + jnp.exp(-x_std)) - log_stds

                    # Tanh correction (CRL line 525)
                    log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                    # Mask unavailable actions: set to 0 (CRL line 526)
                    # Note: This sets log_prob=0 for unavailable, then sums - this is correct per CRL
                    log_prob = jnp.where(avail_mask == 0, 0, log_prob)
                    log_prob = log_prob.sum(-1)  # Sum across action dimensions (CRL line 527)
                
                # Compute Q-value (negative distance) with epsilon for numerical stability
                sa_repr = sa_encoder_apply(critic_params['sa_encoder'], state, action)
                g_repr = goal_encoder_apply(critic_params['goal_encoder'], goal)
                q_value = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1) + 1e-8)
                
                actor_loss = (alpha * log_prob - q_value).mean()
                entropy = -log_prob.mean()
                
                loss_info = {
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return actor_loss, loss_info

            def _alpha_loss_fn(log_alpha, entropy):
                """Temperature loss (matches working implementation exactly)."""
                alpha = jnp.exp(log_alpha)
                # Use stop_gradient to prevent gradient flow back to actor through entropy
                alpha_loss = alpha * jax.lax.stop_gradient(entropy - target_entropy)
                return alpha_loss, {"alpha_loss": alpha_loss, "alpha": alpha}

            # Update critic (both encoders)
            critic_params = {'sa_encoder': params.sa_encoder, 'goal_encoder': params.goal_encoder}
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (critic_loss, critic_info), critic_grads = critic_grad_fn(critic_params, obs, action)
            
            # Average gradients across devices only (not batch axis - handled by minibatching)
            critic_grads, critic_info = lax.pmean((critic_grads, critic_info), axis_name="device")
            
            # Apply critic updates
            critic_updates, new_critic_opt_state = critic_update_fn(
                critic_grads, opt_states.critic
            )
            new_sa_encoder = optax.apply_updates(params.sa_encoder, critic_updates['sa_encoder'])
            new_goal_encoder = optax.apply_updates(params.goal_encoder, critic_updates['goal_encoder'])
            
            # Update actor
            alpha = jnp.exp(params.log_alpha)
            avail_actions = batch_transitions.avail_actions  # Get avail_actions mask from transitions
            # NOTE: future_state is no longer needed since we use relabeled goal from obs
            # But we still pass it to maintain function signature (won't be used)
            future_state = batch_transitions.extras["future_state"]  # Not used anymore, but kept for compatibility
            critic_params = {'sa_encoder': params.sa_encoder, 'goal_encoder': params.goal_encoder}
            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (actor_loss, actor_info), actor_grads = actor_grad_fn(params.actor, critic_params, obs, alpha, avail_actions, future_state, actor_key)
            
            # Average gradients across devices only
            actor_grads, actor_info = lax.pmean((actor_grads, actor_info), axis_name="device")
            
            # Apply actor updates
            actor_updates, new_actor_opt_state = actor_update_fn(
                actor_grads, opt_states.actor
            )
            new_actor = optax.apply_updates(params.actor, actor_updates)
            
            # Update alpha (temperature)
            if config.system.icrl.learnable_temperature:
                alpha_grad_fn = jax.value_and_grad(_alpha_loss_fn, has_aux=True)
                (alpha_loss, alpha_info), alpha_grads = alpha_grad_fn(
                    params.log_alpha, actor_info["entropy"]
                )
                
                # Average gradients across devices only
                alpha_grads, alpha_info = lax.pmean((alpha_grads, alpha_info), axis_name="device")
                
                # Apply alpha updates
                alpha_updates, new_alpha_opt_state = alpha_update_fn(
                    alpha_grads, opt_states.alpha
                )
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
            else:
                new_log_alpha = params.log_alpha
                new_alpha_opt_state = opt_states.alpha
                alpha_info = {"alpha": alpha, "alpha_loss": 0.0}
            
            # Package new params and opt_states
            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
                actor=new_actor,
                log_alpha=new_log_alpha,
            )
            new_opt_states = OptStates(
                actor=new_actor_opt_state,
                critic=new_critic_opt_state,
                alpha=new_alpha_opt_state,
            )
            
            metrics = critic_info | actor_info | alpha_info
            return (new_params, new_opt_states, key), metrics

        # Scan over minibatches
        (params, opt_states, key), train_metrics = jax.lax.scan(
            _update_minibatch, (params, opt_states, key), transitions
        )
        
        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state)
        return learner_state, (episode_metrics, train_metrics)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput:
        """Learner function - performs multiple update steps."""
        # Vectorize over update_batch_size
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")
        learner_state, (episode_metrics, train_metrics) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_metrics,
            train_metrics=train_metrics,
        )

    return learner_fn

def get_prefill_fn(env, buffer, actor_apply, config, get_avail_act):
    n_agents = config.system.num_agents
    num_envs = config.arch.num_envs
    num_envs_agents = num_envs * n_agents
    rollout_len = config.system.rollout_length
    per_actor_step = rollout_len * num_envs_agents  # per *single* update-batch element
    min_replay = getattr(config.system, "explore_steps", 1_000)
    
    def _rollout_once(learner_state):
        """One rollout_length of env steps + push to buffer (no SGD)."""
        params, opt_states, buffer_state, key, env_state = learner_state

        def single_step(carry, _):
            key, env_state, buffer_state = carry
            key, policy_key = jax.random.split(key)

            # Get obs from Brax State (like CRL line 420)
            # Infer num_envs from state shape
            num_envs_actual = env_state.obs.shape[0]
            num_envs_agents = env_state.obs.shape[0] * env_state.obs.shape[1]  # num_envs * num_agents
            obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])  # [N_env*N_agent, obs_dim]
            means, log_stds = actor_apply(params.actor, obs)  # [N_env*N_agent, action_dim]
            
            # Get available actions (like CRL line 434)
            avail_actions = get_avail_act(env_state)  # [N_env, N_agent, action_dim]
            avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])  # [N_env*N_agent, action_dim]
            avail_actions = jnp.asarray(avail_actions, dtype=means.dtype)

            if config.system.discrete_actions:
                means = means - ((1.0 - avail_actions) * 1e10)
                categorical = distrax.Categorical(logits=means)
                action_idx = categorical.sample(seed=policy_key)  # [N_env*N_agent]
                stored_action = jax.nn.one_hot(action_idx, means.shape[-1])  # [N_env*N_agent, action_dim]
            else:
                # GUMBEL-SOFTMAX: Match CRL's approach exactly (lines 426-440)
                stds = jnp.exp(log_stds)  # [N_env*N_agent, action_dim]
                
                # Apply masking before Gumbel sampling (like CRL line 436)
                means = means - ((1.0 - avail_actions) * 1e10)
                
                # Reshape for per-agent noise generation (matching CRL line 428-431)
                means_reshaped = means.reshape(num_envs_actual, n_agents, -1)  # [N_env, N_agent, action_dim]
                means_for_noise = means_reshaped.transpose(1, 0, 2)  # [N_agent, N_env, action_dim]
                stds_reshaped = stds.reshape(num_envs_actual, n_agents, -1)  # [N_env, N_agent, action_dim]
                stds_for_noise = stds_reshaped.transpose(1, 0, 2)  # [N_agent, N_env, action_dim]
                
                # Generate Gumbel noise per agent (matching CRL line 428-431)
                noise_keys = jax.random.split(policy_key, n_agents)
                noise = jax.vmap(
                    lambda rng, m: jax.random.gumbel(
                        rng, shape=(num_envs_actual,) + m.shape[1:], dtype=m.dtype
                    ),
                    in_axes=(0, 0),
                )(noise_keys, means_for_noise)  # [N_agent, N_env, action_dim]
                noise_reshaped = noise.transpose(1, 0, 2).reshape(-1, noise.shape[-1])  # [N_env*N_agent, action_dim]
                
                # Compute continuous actions with tanh (matching CRL line 438)
                stored_action = nn.tanh(means + stds * noise_reshaped)  # [N_env*N_agent, action_dim]
            
            # Reshape actions for env step (like CRL line 451)
            # Reshape stored_action (which is [N_env*N_agent, ...]) to (num_envs, n_agents, ...)
            actions_ = jnp.reshape(stored_action, (-1, n_agents) + stored_action.shape[1:])

            # Step env (returns State, not tuple) - like CRL line 453
            nstate = env.step(env_state, actions_)

            # Extract from Brax State (like CRL lines 456, 462-463)
            flat_obs = obs  # Already flattened
            flat_action = stored_action  # Already [N_env*N_agent, action_dim]
            flat_reward = jnp.repeat(nstate.reward, n_agents)  # Replicate scalar reward
            flat_discount = jnp.repeat(1 - nstate.done, n_agents)  # Compute from done
            flat_avail = avail_actions  # Already flattened

            # Extract extras from state.info (like CRL line 456)
            trunc = jnp.repeat(nstate.info["truncation"], n_agents)  # Replicate scalar
            seed = jnp.repeat(nstate.info["seed"], n_agents)  # Replicate scalar

            transition = ICRLTransition(
                observation=flat_obs,
                action=flat_action,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,  # Store availability mask
                extras={"state_extras": {"truncation": trunc, "seed": seed}},
            )
            return (key, nstate, buffer_state), transition

        (key, env_state, buffer_state), traj = jax.lax.scan(
            single_step, (key, env_state, buffer_state), None, rollout_len
        )
        buffer_state = buffer.insert(buffer_state, traj)

        return LearnerState(params, opt_states, buffer_state, key, env_state)

    # NEW: batched version across update-batch axis (axis 0 of each leaf)
    batched_rollout_once = jax.vmap(_rollout_once, in_axes=0, out_axes=0)


    def prefill_fn(learner_state):
        def filled_enough(ls):
            # buffer_state.size has shape (update_batch, 1) under pmap; take min across that axis.
            size = jnp.min(ls.buffer_state.size)
            return size >= min_replay  # min_replay = config.system.explore_steps

        def cond(ls):
            return jnp.logical_not(filled_enough(ls))

        def body(ls):
            return batched_rollout_once(ls)

        result = jax.lax.while_loop(cond, body, learner_state)
        return result

    return prefill_fn

def learner_setup(
    env: Any, keys: chex.Array, config: DictConfig, get_avail_act: Any
    ) -> Tuple:
    """Initialize learner_fn, networks, optimizers, buffer, and states."""
    # Get available devices
    n_devices = len(jax.devices())
    
    # Get number of agents - should already be set from config
    # (set in run_experiment before wrapping)
    if not hasattr(config.system, 'num_agents') or config.system.num_agents == 0:
        raise ValueError("num_agents must be set in config before calling learner_setup")
    
    # PRNG keys
    key, sa_key, goal_key, actor_key = keys
    
    # Define networks (architectures are fixed: 4x1024 -> 64)
    sa_encoder = SAEncoder()
    goal_encoder = GoalEncoder()
    actor_network = ICRLActor(action_size=env.action_size)
    
    # Initialize network parameters
    n_agents = config.system.num_agents
    
    # Observation dimensions from config
    # Total obs = base(127) + current_health(1) + ultimate_goal(1) = 129
    obs_size = config.system.icrl.obs_dim + config.system.icrl.goal_dim  # 127 + 2 = 129
    # For loss functions: use base + current_health = 128 (excludes ultimate_goal)
    obs_dim = obs_size - 1  # 128
    
    # Create dummy inputs for network initialization  
    init_obs = jnp.zeros((1, obs_size))    # Actor sees full 129-dim observation
    init_state = jnp.zeros((1, obs_dim))   # SA encoder sees 128-dim state
    init_action = jnp.zeros((1, env.action_size))
    goal_dim_actual = config.system.icrl.goal_end_idx - config.system.icrl.goal_start_idx
    init_goal = jnp.zeros((1, goal_dim_actual))
    
    # SA encoder: takes state and action separately
    sa_encoder_params = sa_encoder.init(sa_key, init_state, init_action)
    
    # Goal encoder: takes goal
    goal_encoder_params = goal_encoder.init(goal_key, init_goal)
    
    # Actor: takes full observation
    actor_params = actor_network.init(actor_key, init_obs)
    
    # Initialize log_alpha
    log_alpha = jnp.array(0.0)
    
    # Pack parameters
    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
        actor=actor_params,
        log_alpha=log_alpha,
    )
    
    # Create optimizers (no gradient clipping - matches original ICRL)
    actor_opt = optax.adam(config.system.policy_lr)
    critic_params_struct = {'sa_encoder': sa_encoder_params, 'goal_encoder': goal_encoder_params}
    critic_opt = optax.adam(config.system.q_lr)
    alpha_opt = optax.adam(config.system.alpha_lr)
    
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params_struct)
    alpha_opt_state = alpha_opt.init(log_alpha)
    
    opt_states = OptStates(
        actor=actor_opt_state,
        critic=critic_opt_state,
        alpha=alpha_opt_state,
    )
    
    # Create replay buffer
    num_envs_agents = config.arch.num_envs * n_agents
    # Dummy transition should match the actual transition structure
    dummy_transition = ICRLTransition(
        observation=jnp.zeros((obs_size,)),
        action=jnp.zeros((env.action_size,)),
        reward=0.0,
        discount=0.0,
        avail_actions=jnp.ones((env.action_size,)),  # Default to all actions available
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}},
    )
    
    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer
    
    buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.system.buffer_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=config.system.batch_size,
            num_envs=num_envs_agents,
            episode_length=config.system.episode_length,
        )
    )
    
    # Initialize buffer state
    buffer_state = jax.jit(buffer.init)(key)
    
    # Pack apply and update functions
    apply_fns = (sa_encoder.apply, goal_encoder.apply, actor_network.apply)
    update_fns = (actor_opt.update, critic_opt.update, alpha_opt.update)
    
    # Get learner function and pmap it
    learn = get_learner_fn(env, buffer, apply_fns, update_fns, config, get_avail_act)
    prefill = get_prefill_fn(env, buffer, actor_network.apply, config, get_avail_act)
    learn = jax.pmap(learn, axis_name="device")
    prefill = jax.pmap(prefill, axis_name="device")

    # Initialize environment states (Brax returns just State, not tuple)
    # CRL approach: reset all envs at once, then reshape (like CRL line 286-287)
    # The training wrapper handles vmap internally, so we pass all keys at once
    total_envs = n_devices * config.system.update_batch_size * config.arch.num_envs
    key, env_key = jax.random.split(key)
    env_keys = jax.random.split(env_key, total_envs)  # (total_envs, 2)
    
    # Reset all environments at once (wrapper handles vmap)
    env_states = env.reset(env_keys)  # Returns state with obs shape (total_envs, num_agents, obs_dim)
    
    # Reshape states to (n_devices, update_batch_size, num_envs, ...)
    # The state is a PyTree, so we need to reshape each leaf
    def reshape_state_leaf(x):
        """Reshape a state leaf from (total_envs, ...) to (n_devices, update_batch_size, num_envs, ...)"""
        if isinstance(x, (jnp.ndarray, chex.Array)) and x.ndim > 0:
            # Reshape the first dimension
            new_shape = (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
            return x.reshape(new_shape)
        return x
    
    env_states = tree.map(reshape_state_leaf, env_states)
    
    # Load model from checkpoint if specified
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params
    
    # Replicate learner state across devices and batches
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.system.update_batch_size)
    step_keys = jnp.array(step_keys).reshape(n_devices, config.system.update_batch_size, -1)
    
    # Replicate params, opt_states, buffer_state (but NOT step_keys - already shaped)
    replicate_items = (params, opt_states, buffer_state)
    
    # Duplicate for update_batch_size
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_items = tree.map(broadcast, replicate_items)
    
    # Duplicate across devices
    replicate_items = flax.jax_utils.replicate(replicate_items, devices=jax.devices())
    
    # Unpack replicated items
    params, opt_states, buffer_state = replicate_items
    init_learner_state = LearnerState(params, opt_states, buffer_state, step_keys, env_states)
    

    return learn, prefill, actor_network, init_learner_state

def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "ff_icrl"
    config = copy.deepcopy(_config)
    
    n_devices = len(jax.devices())
    
    # Create environments using CRL method (like CRL lines 261-288)
    from mava.systems.crl.envs.smax import SmaxEnv
    
    # Create base environment
    base_env = SmaxEnv(map_name=config.env.scenario.task_name)
    
    # Extract environment parameters BEFORE wrapping (like CRL lines 266-272)
    config.system.num_agents = base_env.env.num_agents
    
    # Wrap with Brax training wrapper
    env = envs.training.wrap(
        base_env,
        episode_length=config.system.episode_length,
    )
    eval_env = envs.training.wrap(
        SmaxEnv(map_name=config.env.scenario.task_name),
        episode_length=config.system.episode_length,
    )
    
    # JIT the step function (like CRL line 288)
    env.step = jax.jit(env.step)
    eval_env.step = jax.jit(eval_env.step)
    
    # Extract remaining environment parameters (like CRL lines 266-272)
    # Note: env.observation_size is 129, so obs_dim for loss functions is 128
    # But we keep config.system.icrl.obs_dim as 127 (base) for network init
    # The loss functions use observation_size - 1 = 128 internally
    config.system.icrl.goal_start_idx = env.observation_size - 2  # 127
    config.system.icrl.goal_end_idx = env.observation_size - 1    # 128
    
    # Get available actions function (like CRL line 277)
    get_avail_act = jax.vmap(env.get_avail_actions)
    
    # PRNG keys
    key, key_e, sa_key, goal_key, actor_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=5
    )
    
    learn, prefill, actor_network, learner_state = learner_setup(
        env, (key, sa_key, goal_key, actor_key), config, get_avail_act
    )

    learner_state = prefill(learner_state)
    jax.block_until_ready(learner_state)
    
    # Setup evaluator using CRL style (like CRL lines 692-699)
    # Create deterministic actor step function for evaluation
    def deterministic_actor_step(training_state, env, env_state, extra_fields=()):
        """Deterministic actor step for evaluation (matches CRL line 386-413)."""
        # training_state is actually just the actor params (we'll adapt the interface)
        actor_params = training_state
        
        # Get obs from Brax State (observation already includes goal at index 128)
        # First flatten: [N_env, N_agent, obs_dim] -> [N_env*N_agent, obs_dim]
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])  # [N_env*N_agent, obs_dim+goal_dim=129]
        num_envs_actual = env_state.obs.shape[0]
        n_agents = config.system.num_agents
        
        # For evaluation, we use the ultimate goal (0 = all enemies dead)
        # The environment observation already has goal=0 at index 128, so we can use it directly
        # IMPORTANT: Must match training's input shape (line 94-95 in _env_step)
        # Training reshapes to (N_env, N_agent, obs_dim) before passing to actor
        # The network was initialized with this shape, so we must use it here too
        obs_reshaped = obs.reshape(num_envs_actual, n_agents, -1)  # [N_env, N_agent, obs_dim]
        means, _ = actor_network.apply(actor_params, obs_reshaped)  # Only need means for deterministic
        # means shape: [N_env, N_agent, action_dim] - already in the right shape!
        
        # Get available actions (like CRL line 394)
        # Note: In CRL, get_avail_act is created from training env but used with eval env_state
        # This works because both use the same underlying SmaxEnv structure
        avail_actions = get_avail_act(env_state)  # [N_env, N_agent, action_dim]
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])  # [N_env*N_agent, action_dim]
        avail_actions = jnp.asarray(avail_actions, dtype=means.dtype)
        
        # Apply masking and get deterministic actions (tanh of means, no noise)
        # means is already [N_env, N_agent, action_dim], flatten it to match avail_actions
        means_flat = means.reshape(-1, means.shape[-1])  # [N_env*N_agent, action_dim]
        means_flat = means_flat - (1.0 - avail_actions) * 1e10  # Mask unavailable actions
        actions = nn.tanh(means_flat)  # Deterministic: just tanh(means), no Gumbel noise
        # actions shape: [N_env*N_agent, action_dim]
        
        # Reshape actions for env.step: [N_env, N_agent, action_dim]
        actions_ = actions.reshape(num_envs_actual, n_agents, -1)  # [N_env, N_agent, action_dim]
        
        # Step environment
        nstate = env.step(env_state, actions_)
        
        # Create transition (not really used, but needed for interface)
        # Note: actions is already flattened [N_env*N_agent, action_dim], matching obs shape
        from mava.systems.icrl.types import Transition as ICRLTransition
        state_extras = {x: jnp.repeat(nstate.info[x], n_agents) for x in extra_fields}
        transition = ICRLTransition(
            observation=obs,  # [N_env*N_agent, obs_dim]
            action=actions,  # [N_env*N_agent, action_dim] (already flattened)
            reward=jnp.repeat(nstate.reward, n_agents),  # [N_env*N_agent]
            discount=jnp.repeat(1 - nstate.done, n_agents),  # [N_env*N_agent]
            avail_actions=avail_actions,  # [N_env*N_agent, action_dim]
            extras={"state_extras": state_extras},
        )
        
        return nstate, transition
    
    # Create evaluator (like CRL line 693-699)
    key_e, eval_env_key = jax.random.split(key_e, 2)
    # Use num_eval_episodes as num_eval_envs (same concept - parallel eval environments)
    num_eval_envs = getattr(config.arch, 'num_eval_episodes', 32)
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=num_eval_envs,
        episode_length=config.system.episode_length,
        key=eval_env_key,
    )
    
    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates must be greater than number of evaluations."
    
    # Calculate number of updates per evaluation
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )
    
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    
    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )
    
    max_episode_return = -jnp.inf
    best_params = None
    
    for eval_step in range(config.arch.num_evaluation):
        # Train
        start_time = time.time()
        
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)
        
        # Log the results of training
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        
        # Extract episode metrics if available
        episode_metrics = learner_output.episode_metrics
        if episode_metrics:
            episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time
        
        # Log timesteps and metrics
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if episode_metrics:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)
        
        # Prepare for evaluation (like CRL line 724)
        # Extract actor params (fully unreplicated - remove device AND batch dimensions)
        trained_params = unreplicate_n_dims(learner_state.params.actor)
        
        # Create a simple wrapper to match CRL's training_state interface
        # The evaluator expects training_state.actor_state.params, but we just pass params directly
        # The deterministic_actor_step function handles this by treating training_state as params
        training_state_wrapper = trained_params
        
        # Aggregate training metrics for evaluator (like CRL line 709-721)
        # train_metrics is already aggregated from scan, but may need flattening
        train_metrics_dict = {}
        if learner_output.train_metrics:
            # Average metrics (like CRL line 709: jax.tree_util.tree_map(jnp.mean, metrics))
            # train_metrics may be a dict of arrays, need to convert to scalars
            def extract_metric_value(v):
                """Extract scalar value from metric (handles arrays, JAX arrays, etc.)"""
                # First, convert to JAX array to get consistent interface
                try:
                    v_array = jnp.array(v)
                except (TypeError, ValueError):
                    # If conversion fails, try direct conversion
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return 0.0
                
                # Check size: if size > 1, compute mean; if size == 1, extract scalar
                if v_array.size == 0:
                    return 0.0
                elif v_array.size == 1:
                    # Single element: extract scalar
                    return float(v_array.item())
                else:
                    # Multiple elements: compute mean
                    return float(jnp.mean(v_array))
            
            for key, value in learner_output.train_metrics.items():
                train_metrics_dict[f"training/{key}"] = extract_metric_value(value)
        
        # Run evaluation (like CRL line 724)
        # Note: evaluator.run_evaluation expects training_state and training_metrics
        # Add training metrics that CRL expects (like CRL line 716-721)
        train_metrics_dict["training/sps"] = steps_per_rollout / elapsed_time
        train_metrics_dict["training/walltime"] = elapsed_time * (eval_step + 1)  # Approximate walltime
        # Extract envsteps from learner_state (if available)
        # Note: ICRL doesn't track envsteps the same way, so we approximate
        train_metrics_dict["training/envsteps"] = float(t)  # Use timestep as proxy
        
        try:
            eval_metrics = evaluator.run_evaluation(training_state_wrapper, train_metrics_dict)
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            eval_metrics = {}
        
        # Log evaluation metrics
        if eval_metrics:
            # Remove "eval/" prefix from keys so they're treated as main metrics by NeptuneLogger
            # (NeptuneLogger filters out non-main metrics when detailed_logging=False)
            # Main metrics are those without "/" or ending with "/mean"
            # Since LogEvent.EVAL already provides context, we don't need "eval/" prefix
            eval_metrics_for_logging = {}
            for key, value in eval_metrics.items():
                # Remove "eval/" prefix if present
                if key.startswith("eval/"):
                    new_key = key[5:]  # Remove "eval/" prefix (5 characters)
                else:
                    new_key = key
                eval_metrics_for_logging[new_key] = value
            
            logger.log(eval_metrics_for_logging, t, eval_step, LogEvent.EVAL)
            
            # Extract episode return from metrics (CRL uses episode_success_any for win rate)
            # Use original keys for extraction
            if config.env.eval_metric in eval_metrics:
                episode_return = float(eval_metrics[config.env.eval_metric])
            elif "eval/episode_success_any" in eval_metrics:
                episode_return = float(eval_metrics["eval/episode_success_any"])
            elif "episode_success_any" in eval_metrics:
                episode_return = float(eval_metrics["episode_success_any"])
            elif "eval/episode_reward" in eval_metrics:
                episode_return = float(jnp.mean(eval_metrics["eval/episode_reward"]))
            elif "episode_reward" in eval_metrics:
                episode_return = float(jnp.mean(eval_metrics["episode_reward"]))
            else:
                episode_return = 0.0
        else:
            episode_return = 0.0
        
        if save_checkpoint:
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )
        
        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return
        
        # Update learner state
        learner_state = learner_output.learner_state
    
    # Record final performance
    if eval_metrics and config.env.eval_metric in eval_metrics:
        eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))
    else:
        eval_performance = 0.0
    
    # Measure absolute metric (disabled for now)
    # if config.arch.absolute_metric:
    #     abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
    #     eval_keys = jax.random.split(key, n_devices)
    #     eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
    #     t = int(steps_per_rollout * (eval_step + 1))
    #     logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
    
    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_icrl.yaml",
    version_base="1.2",
    )
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes
    OmegaConf.set_struct(cfg, False)
    
    # Run experiment
    eval_performance = run_experiment(cfg)
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
