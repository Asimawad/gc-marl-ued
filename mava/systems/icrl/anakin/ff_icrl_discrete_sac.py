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
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import DiscreteSAEncoder, GoalEncoder, DiscreteICRLActor
from mava.systems.icrl.types import ICRLParams, OptStates, LearnerState, Transition as ICRLTransition
from mava.types import MarlEnv, ExperimentOutput
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue
from hydra.utils import instantiate
from mava.networks.torsos import MLPTorso

def get_learner_fn(
    env: MarlEnv,
    buffer: TrajectoryUniformSamplingQueue,
    apply_fns: Tuple,
    update_fns: Tuple,
    config: DictConfig,
) -> Any:
    """Get the learner function for discrete SAC."""
    # Unpack apply and update functions
    sa_encoder_apply, goal_encoder_apply, actor_apply = apply_fns
    actor_update_fn, critic_update_fn, alpha_update_fn = update_fns

    # Multi-agent dimensions
    n_agents = env.num_agents
    action_dim = env.action_dim
    num_envs_agents = config.arch.num_envs * n_agents

    base_obs_dim = config.system.icrl.obs_dim  # From environment
    goal_dim = config.system.icrl.goal_dim  # Always 2
    
    # For loss functions: use base + current_progress (excludes ultimate_goal)
    obs_dim = base_obs_dim + 1  # base + current_progress
    
    # For hindsight relabeling: extract only current_progress (first dim of goal)
    goal_start_idx = config.system.icrl.goal_start_idx  # = base_obs_dim
    goal_end_idx = config.system.icrl.goal_end_idx  # = base_obs_dim + 1

    # Target entropy for discrete SAC
    # We use adaptive target entropy based on available actions
    # This accounts for action masking - target adjusts to number of actually available actions
    target_entropy_scale = getattr(config.system, 'target_entropy_scale', 0.89)
    use_adaptive_target_entropy = getattr(config.system, 'use_adaptive_target_entropy', True)
    
    # Fallback static target (used if adaptive is disabled)
    static_target_entropy = target_entropy_scale * jnp.log(action_dim)
    
    # Logsumexp penalty coefficient
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff

    def _env_step(
        learner_state: LearnerState, _: Any
        ) -> Tuple[LearnerState, ICRLTransition]:
        """Step the environment for rollout_length steps."""
        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        def single_step(carry, _):
            """Single environment step with discrete actions."""
            key, env_state, last_timestep, buffer_state = carry

            # RNG and obs
            key, policy_key = jax.random.split(key)
            obs = last_timestep.observation.agents_view  # [N_env, N_agent, obs_dim]
            
            # Get action mask
            avail_actions = None
            if hasattr(env, 'action_mask'):
                if hasattr(env_state, 'env_state'):
                    wrapped_env_states = env_state.env_state
                    if hasattr(wrapped_env_states, 'state'):
                        wrapped_env_states = wrapped_env_states.state
                else:
                    wrapped_env_states = env_state
                avail_actions = jax.vmap(env.action_mask)(wrapped_env_states)
            else:
                avail_actions = jnp.ones(obs.shape[:-1] + (action_dim,), dtype=jnp.bool_)

            # Get action distribution from actor
            policy = actor_apply(params.actor, obs, avail_actions)
            action_indices = policy.sample(seed=policy_key)
            actions_onehot = jax.nn.one_hot(action_indices, action_dim)

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action_indices)

            # Flatten for buffer
            flat_obs = last_timestep.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = actions_onehot.reshape(num_envs_agents, -1)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = avail_actions.reshape(num_envs_agents, -1)

            # Broadcast seed and truncation to all agents
            trunc_per_env = timestep.extras.get("truncation", jnp.zeros(config.arch.num_envs, dtype=jnp.float32))
            seed_per_env = timestep.extras.get("seed", jnp.zeros(config.arch.num_envs, dtype=jnp.float32))
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)

            transition = ICRLTransition(
                observation=flat_obs,
                action=flat_action,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                extras={"state_extras": {"truncation": trunc, "seed": seed}},
            )

            return (key, env_state, timestep, buffer_state), transition

        # Collect rollout_length transitions
        (key, env_state, last_timestep, buffer_state), traj_batch = jax.lax.scan(
            single_step, (key, env_state, last_timestep, buffer_state), None, config.system.rollout_length
        )
        # Add trajectory to buffer (time-major format)
        buffer_state = buffer.insert(buffer_state, traj_batch)
        
        # Get episode metrics
        metrics = last_timestep.extras.get("episode_metrics", {})
        
        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
        return learner_state, metrics

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network (collect rollout + train)."""
        
        # Collect experience
        learner_state, episode_metrics = _env_step(learner_state, None)

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state
        
        # Sample batch from buffer
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
        
        # Reshape into batches of batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
            transitions,
        )

        def _update_minibatch(carry, batch_transitions):
            """Update networks on a single minibatch using discrete SAC."""
            params, opt_states, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)
            
            # Extract data for this batch
            obs = batch_transitions.observation  # [batch_size, obs_dim+goal_dim]
            action = batch_transitions.action    # [batch_size, action_dim] (one-hot)
            avail_actions = batch_transitions.avail_actions
            future_state = batch_transitions.extras["future_state"]

            def _contrastive_loss_fn(critic_params, obs, action):
                """InfoNCE contrastive loss for critic (same as continuous version)."""
                # Split observation into state and goal
                state = obs[:, :obs_dim]
                goal = obs[:, obs_dim:]
                
                # For contrastive loss, we need SA embeddings for the TAKEN actions
                # DiscreteSAEncoder outputs [batch, num_actions, emb_dim]
                sa_repr_all = sa_encoder_apply(critic_params['sa_encoder'], state)
                
                # Extract embedding for the taken action using one-hot indexing
                # action is [batch, action_dim] one-hot
                # sa_repr_all is [batch, action_dim, emb_dim]
                sa_repr = jnp.einsum('ba,bae->be', action, sa_repr_all)  # [batch, emb_dim]
                
                g_repr = goal_encoder_apply(critic_params['goal_encoder'], goal)
                
                # InfoNCE: compute pairwise distances
                logits = -jnp.sqrt(
                    jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1) + 1e-8
                )  # [batch, batch]
                
                # InfoNCE loss: maximize diagonal (positive pairs), minimize off-diagonal
                critic_loss = -jnp.mean(
                    jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1)
                )
                
                # Logsumexp regularization
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

            def _actor_loss_fn(actor_params, critic_params, alpha, key):
                """Actor loss for discrete SAC.
                
                Key difference from continuous: computes EXACT expectation over all actions
                Loss = E_s [sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a))]
                """
                state = obs[:, :obs_dim]
                goal = future_state[:, goal_start_idx:goal_end_idx]
                observation = jnp.concatenate([state, goal], axis=1)

                # Get action distribution
                policy = actor_apply(actor_params, observation, avail_actions)
                logits = policy.distribution.logits
                probs = jax.nn.softmax(logits)
                log_probs = jax.nn.log_softmax(logits)
                
                # Clip log probs for numerical stability (handles -inf from masking)
                log_probs = jnp.clip(log_probs, a_min=-20.0)

                # Compute Q-values for ALL actions at once
                # DiscreteSAEncoder outputs [batch, num_actions, emb_dim]
                sa_repr_all = sa_encoder_apply(critic_params['sa_encoder'], state)
                g_repr = goal_encoder_apply(critic_params['goal_encoder'], goal)
                
                # Q-value for each action: -distance(sa_repr[a], g_repr)
                # sa_repr_all: [batch, num_actions, emb_dim]
                # g_repr: [batch, emb_dim] -> [batch, 1, emb_dim] for broadcasting
                q_values = -jnp.sqrt(
                    jnp.sum((sa_repr_all - g_repr[:, None, :]) ** 2, axis=-1) + 1e-8
                )  # [batch, num_actions]

                # Discrete SAC actor loss: exact expectation over all actions
                actor_loss = (probs * (alpha * log_probs - q_values)).sum(axis=-1).mean()
                entropy = -(probs * log_probs).sum(axis=-1).mean()
                
                # Compute adaptive target entropy based on available actions
                if use_adaptive_target_entropy:
                    # Count available actions per sample (avail_actions is boolean mask)
                    num_available = avail_actions.sum(axis=-1).astype(jnp.float32)  # [batch]
                    # Clamp to avoid log(0) or log(1) edge cases
                    num_available = jnp.clip(num_available, a_min=2.0, a_max=action_dim)
                    # Target entropy per sample: scale * log(num_available_actions)
                    target_entropy_per_sample = target_entropy_scale * jnp.log(num_available)
                    # Average over batch
                    adaptive_target_entropy = target_entropy_per_sample.mean()
                else:
                    adaptive_target_entropy = static_target_entropy

                return actor_loss, {
                    "actor_loss": actor_loss, 
                    "entropy": entropy,
                    "target_entropy": adaptive_target_entropy,
                    "num_available_actions": avail_actions.sum(axis=-1).mean() if use_adaptive_target_entropy else action_dim,
                }

            def _alpha_loss_fn(log_alpha, entropy, target_entropy_dynamic):
                """Temperature (alpha) loss for discrete SAC.
                
                Loss = alpha * (entropy - target_entropy)
                - If entropy < target: loss negative -> alpha increases -> more exploration
                - If entropy > target: loss positive -> alpha decreases -> more exploitation
                
                Args:
                    log_alpha: Log of temperature parameter
                    entropy: Current policy entropy
                    target_entropy_dynamic: Adaptive target entropy based on available actions
                """
                alpha = jnp.exp(log_alpha)
                # Use stop_gradient to prevent gradient flow back to actor through entropy and target
                alpha_loss = alpha * jax.lax.stop_gradient(entropy - target_entropy_dynamic)
                return alpha_loss, {
                    "alpha_loss": alpha_loss,
                    "alpha": alpha,
                    "log_alpha": log_alpha,
                }

            # Update critic
            critic_params = {'sa_encoder': params.sa_encoder, 'goal_encoder': params.goal_encoder}
            critic_grad_fn = jax.value_and_grad(_contrastive_loss_fn, has_aux=True)
            (critic_loss, critic_info), critic_grads = critic_grad_fn(critic_params, obs, action)
            critic_grads, critic_info = lax.pmean((critic_grads, critic_info), axis_name="device")
            
            critic_updates, new_critic_opt_state = critic_update_fn(critic_grads, opt_states.critic)
            new_sa_encoder = optax.apply_updates(params.sa_encoder, critic_updates['sa_encoder'])
            new_goal_encoder = optax.apply_updates(params.goal_encoder, critic_updates['goal_encoder'])

            # Update actor
            alpha = jnp.exp(params.log_alpha)
            critic_params = {'sa_encoder': params.sa_encoder, 'goal_encoder': params.goal_encoder}
            
            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (actor_loss, actor_info), actor_grads = actor_grad_fn(
                params.actor, critic_params, alpha, actor_key
            )
            actor_grads, actor_info = lax.pmean((actor_grads, actor_info), axis_name="device")
            
            actor_updates, new_actor_opt_state = actor_update_fn(actor_grads, opt_states.actor)
            new_actor = optax.apply_updates(params.actor, actor_updates)

            # Update alpha (temperature parameter)
            if config.system.autotune:
                alpha_grad_fn = jax.value_and_grad(_alpha_loss_fn, has_aux=True)
                (alpha_loss, alpha_info), alpha_grads = alpha_grad_fn(
                    params.log_alpha, actor_info["entropy"], actor_info["target_entropy"]
                )
                alpha_grads, alpha_info = lax.pmean((alpha_grads, alpha_info), axis_name="device")
                
                alpha_updates, new_alpha_opt_state = alpha_update_fn(alpha_grads, opt_states.alpha)
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
                
                # Clip log_alpha to prevent explosion
                new_log_alpha = jnp.clip(new_log_alpha, a_min=-10.0, a_max=2.0)
            else:
                new_log_alpha = params.log_alpha
                new_alpha_opt_state = opt_states.alpha
                alpha_info = {"alpha": alpha, "alpha_loss": 0.0}
            
            # Add target_entropy and num_available_actions to metrics from actor_info
            alpha_info["target_entropy"] = actor_info.get("target_entropy", static_target_entropy)
            alpha_info["num_available_actions"] = actor_info.get("num_available_actions", action_dim)
            
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
        
        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
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


def get_prefill_fn(env, buffer, actor_apply, config):
    """Prefill function for discrete SAC."""
    n_agents = env.num_agents
    num_envs = config.arch.num_envs
    num_envs_agents = num_envs * n_agents
    rollout_len = config.system.rollout_length
    action_dim = env.action_dim
    min_replay = getattr(config.system, "explore_steps", 1_000)

    def _rollout_once(learner_state):
        """One rollout_length of env steps + push to buffer (no SGD)."""
        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        def single_step(carry, _):
            key, env_state, last_ts, buffer_state = carry
            key, policy_key = jax.random.split(key)

            obs = last_ts.observation.agents_view  # [N_env, N_agent, obs_dim]
            
            # Get available actions mask
            avail = None
            if hasattr(env, "action_mask"):
                wrapped = env_state
                if hasattr(wrapped, "env_state"):
                    wrapped = wrapped.env_state
                    if hasattr(wrapped, "state"):
                        wrapped = wrapped.state
                avail = jax.vmap(env.action_mask)(wrapped)  # [N_env, N_agent, action_dim]
            else:
                avail = jnp.ones(obs.shape[:-1] + (action_dim,), dtype=jnp.bool_)

            # Get action from discrete policy
            policy = actor_apply(params.actor, obs, avail)
            action_idx = policy.sample(seed=policy_key)
            stored_action = jax.nn.one_hot(action_idx, action_dim)
            env_actions = action_idx

            env_state, ts = jax.vmap(env.step, in_axes=(0, 0))(env_state, env_actions)

            flat_obs = last_ts.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = stored_action.reshape(num_envs_agents, -1)
            flat_reward = ts.reward.reshape(num_envs_agents)
            flat_discount = ts.discount.reshape(num_envs_agents)
            flat_avail = avail.reshape(num_envs_agents, -1)

            trunc_per_env = ts.extras.get("truncation", jnp.zeros(num_envs, dtype=jnp.float32))
            seed_per_env = ts.extras.get("seed", jnp.zeros(num_envs, dtype=jnp.float32))
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)

            transition = ICRLTransition(
                observation=flat_obs,
                action=flat_action,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                extras={"state_extras": {"truncation": trunc, "seed": seed}},
            )
            return (key, env_state, ts, buffer_state), transition

        (key, env_state, last_timestep, buffer_state), traj = jax.lax.scan(
            single_step, (key, env_state, last_timestep, buffer_state), None, rollout_len
        )
        buffer_state = buffer.insert(buffer_state, traj)

        return LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)

    # Batched version across update-batch axis
    batched_rollout_once = jax.vmap(_rollout_once, in_axes=0, out_axes=0)

    def prefill_fn(learner_state):
        def filled_enough(ls):
            size = jnp.min(ls.buffer_state.size)
            return size >= min_replay

        def cond(ls):
            return jnp.logical_not(filled_enough(ls))

        def body(ls):
            return batched_rollout_once(ls)

        result = jax.lax.while_loop(cond, body, learner_state)
        return result

    return prefill_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple:
    """Initialize learner_fn, networks, optimizers, buffer, and states for discrete SAC."""
    # Get available devices
    n_devices = len(jax.devices())
    # Get number of agents
    config.system.num_agents = env.num_agents
    
    # PRNG keys
    key, sa_key, goal_key, actor_key = keys
    
    # Define networks with configurable torsos from config
    
    
    # Instantiate torsos from config (with fallback to default if not specified)
    if hasattr(config.network, 'sa_encoder_network') and hasattr(config.network.sa_encoder_network, 'pre_torso'):
        sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    else:
        sa_encoder_torso = MLPTorso(layer_sizes=[1024, 1024, 1024, 1024], activation="swish", use_layer_norm=True)
    
    if hasattr(config.network, 'goal_encoder_network') and hasattr(config.network.goal_encoder_network, 'pre_torso'):
        goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)
    else:
        goal_encoder_torso = MLPTorso(layer_sizes=[1024, 1024, 1024, 1024], activation="swish", use_layer_norm=True)
    
    if hasattr(config.network, 'actor_network') and hasattr(config.network.actor_network, 'pre_torso'):
        actor_torso = instantiate(config.network.actor_network.pre_torso)
    else:
        actor_torso = MLPTorso(layer_sizes=[1024, 1024, 1024, 1024], activation="swish", use_layer_norm=True)
    
    # Create networks with configurable torsos
    # Key difference: DiscreteSAEncoder takes only state and outputs [batch, num_actions, emb_dim]
    sa_encoder = DiscreteSAEncoder(torso=sa_encoder_torso, num_actions=env.action_dim, emb_dim=64)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, emb_dim=64)
    actor_network = DiscreteICRLActor(torso=actor_torso, action_size=env.action_dim)
    
    # Initialize network parameters
    n_agents = env.num_agents
    
    # Observation dimensions from config
    base_obs_dim = config.system.icrl.obs_dim
    goal_dim = config.system.icrl.goal_dim  # Always 2
    obs_size = base_obs_dim + goal_dim  # Total observation size
    
    # For loss functions: use base + current_progress (excludes ultimate_goal)
    obs_dim = base_obs_dim + 1
    
    # Create dummy inputs for network initialization
    init_obs = jnp.zeros((1, obs_size))    # Actor sees full observation
    init_state = jnp.zeros((1, obs_dim))   # SA encoder sees state only
    init_action_mask = jnp.ones((1, env.action_dim), dtype=jnp.bool_)
    goal_dim_actual = config.system.icrl.goal_end_idx - config.system.icrl.goal_start_idx
    init_goal = jnp.zeros((1, goal_dim_actual))
    
    # SA encoder: takes state only (outputs embeddings for all actions)
    sa_encoder_params = sa_encoder.init(sa_key, init_state)
    
    # Goal encoder: takes goal
    goal_encoder_params = goal_encoder.init(goal_key, init_goal)
    
    # Actor: takes full observation and action mask
    actor_params = actor_network.init(actor_key, init_obs, init_action_mask)

    if config.system.autotune:
        log_alpha = jnp.array(0.0)
    else:
        alpha = jnp.log(config.system.init_alpha)
        log_alpha = jnp.array(alpha)
    
    # Pack parameters
    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
        actor=actor_params,
        log_alpha=log_alpha,
    )
    
    # Make opt states
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    actor_opt = optax.chain(grad_clip, optax.adam(config.system.policy_lr))
    critic_opt = optax.chain(grad_clip, optax.adam(config.system.q_lr))
    alpha_opt = optax.chain(grad_clip, optax.adam(config.system.alpha_lr))
    
    critic_params_struct = {'sa_encoder': sa_encoder_params, 'goal_encoder': goal_encoder_params}
    
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
    dummy_transition = ICRLTransition(
        observation=jnp.zeros((obs_size,)),
        action=jnp.zeros((env.action_dim,)),
        reward=0.0,
        discount=0.0,
        avail_actions=jnp.ones((env.action_dim,)),
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
    learn = get_learner_fn(env, buffer, apply_fns, update_fns, config)
    prefill = get_prefill_fn(env, buffer, actor_network.apply, config)
    learn = jax.pmap(learn, axis_name="device")
    prefill = jax.pmap(prefill, axis_name="device")

    # Initialize environment states and timesteps
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)
    
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
    
    # Replicate params, opt_states, buffer_state
    replicate_items = (params, opt_states, buffer_state)
    
    # Duplicate for update_batch_size
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_items = tree.map(broadcast, replicate_items)
    
    # Duplicate across devices
    replicate_items = flax.jax_utils.replicate(replicate_items, devices=jax.devices())
    
    # Unpack replicated items
    params, opt_states, buffer_state = replicate_items
    init_learner_state = LearnerState(params, opt_states, buffer_state, step_keys, env_states, timesteps)

    return learn, prefill, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "ff_icrl_discrete_sac"
    config = copy.deepcopy(_config)
    
    # Set flag to enable discrete ICRL actor handling in evaluator
    config.system.use_discrete_icrl = True
    
    n_devices = len(jax.devices())
    
    # Create environments for train and eval
    env, eval_env = environments.make(config)
    
    # PRNG keys
    key, key_e, sa_key, goal_key, actor_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=5
    )
    
    learn, prefill, actor_network, learner_state = learner_setup(
        env, (key, sa_key, goal_key, actor_key), config
    )

    learner_state = prefill(learner_state)
    jax.block_until_ready(learner_state)
    
    # Setup evaluator
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
    
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
        
        # Prepare for evaluation
        trained_params = unreplicate_batch_dim(learner_state.params.actor)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])
        
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
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))
    
    # Measure absolute metric
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)
        
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
        
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
    
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

