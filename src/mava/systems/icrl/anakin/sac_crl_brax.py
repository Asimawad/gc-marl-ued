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
from typing import Any, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from brax import envs
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from hydra.utils import instantiate
from jax import tree
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

# Mava imports
from mava.evaluator import CrlEvaluator
from mava.wrappers.smax import SmaxEnv
from mava.systems.icrl.types import Transition, ICRLParams, OptStates
from mava.types import ExperimentOutput
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue, ReplayBufferState

import flax.linen as nn


# Network definitions
class SA_encoder(nn.Module):
    """State-Action encoder with LayerNorm."""
    
    rep_size: int = 64
    norm_type: str = "layer_norm"
    
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        x = jnp.concatenate([s, a], axis=-1)
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


class G_encoder(nn.Module):
    """Goal encoder with LayerNorm."""
    
    rep_size: int = 64
    norm_type: str = "layer_norm"
    
    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(g)
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


class Actor(nn.Module):
    """Actor network for SAC."""
    
    action_size: int
    norm_type: str = "layer_norm"
    
    LOG_STD_MAX = 5
    LOG_STD_MIN = -5
    
    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
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
        
        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        
        return mean, log_std


# Learner state definition
class SACCRLParams(NamedTuple):
    """Parameters for SAC-CRL."""
    actor: Any
    sa_encoder: Any
    goal_encoder: Any


class SACOptStates(NamedTuple):
    """Optimizer states for SAC-CRL."""
    actor: Any
    critic: Any
    alpha: Any


class SACCRLLearnerState(NamedTuple):
    """Learner state for SAC-CRL with Brax environments."""
    
    params: SACCRLParams
    opt_states: SACOptStates
    alpha_params: Dict
    key: chex.PRNGKey
    env_state: Any
    buffer_state: ReplayBufferState
    t: chex.Array


def flatten_crl_fn(buffer_config, transition, sample_key):
    """HER-style goal relabeling (same as PQN-CRL)."""
    
    gamma, obs_dim, goal_start_idx, goal_end_idx = buffer_config
    
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount
    
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
    )
    
    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    
    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_start_idx : goal_end_idx]
    future_state = future_state[:, : obs_dim]
    state = transition.observation[:-1, : obs_dim]
    new_obs = jnp.concatenate([state, goal], axis=1)
    
    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "seed": jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
    }
    
    return transition._replace(
        observation=jnp.squeeze(new_obs),
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        avail_actions=jnp.squeeze(transition.avail_actions[:-1]),
        extras=extras,
    )


def get_learner_fn(
    env,
    actor: nn.Module,
    sa_encoder: nn.Module,
    g_encoder: nn.Module,
    actor_update_fn,
    critic_update_fn,
    alpha_update_fn,
    replay_buffer: TrajectoryUniformSamplingQueue,
    config: DictConfig,
) -> Any:
    """Get the learner function for SAC-CRL with Brax infrastructure."""
    
    # Extract config values
    n_agents = config.system.num_agents
    action_size = config.system.action_size
    num_envs = config.arch.num_envs
    num_envs_agents = num_envs * n_agents
    rollout_length = config.system.rollout_length
    batch_size = config.system.batch_size
    gamma = config.system.gamma
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    discrete_actions = config.system.discrete_actions
    target_entropy = config.system.target_entropy
    
    # Observation dimensions
    obs_dim = config.system.obs_dim
    goal_start_idx = config.system.goal_start_idx
    goal_end_idx = config.system.goal_end_idx
    
    # Buffer config tuple
    buffer_config = (gamma, obs_dim, goal_start_idx, goal_end_idx)
    
    # Get available actions function
    get_avail_act = jax.vmap(env.get_avail_actions)
    
    def actor_step(actor_params, env, env_state, key, extra_fields):
        """Collect experience using current policy."""
        keys = jax.random.split(key, n_agents)
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        means, log_stds = actor.apply(actor_params, obs)
        
        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        
        if not discrete_actions:
            # Continuous actions with Gumbel noise
            stds = jnp.exp(log_stds)
            noise = jax.vmap(lambda rng: jax.random.gumbel(
                rng, shape=(num_envs,) + means.shape[1:], dtype=means.dtype
            ), in_axes=0, out_axes=1)(keys)
            noise_ = jnp.reshape(noise, (-1,) + noise.shape[2:])
            
            means = means - ((1 - avail_actions) * 1e10)
            actions = nn.tanh(means + stds * noise_)
            transition_actions = actions
        else:
            # Discrete actions
            action_logits = means - ((1 - avail_actions) * 1e10)
            pi = distrax.Categorical(logits=action_logits)
            actions = pi.sample(seed=keys[0])
            transition_actions = jax.nn.one_hot(actions, means.shape[1])
        
        actions_ = jnp.reshape(actions, (-1, n_agents) + actions.shape[1:])
        nstate = env.step(env_state, actions_)
        
        state_extras = {x: jnp.repeat(nstate.info[x], n_agents) for x in extra_fields}
        
        return nstate, Transition(
            observation=obs,
            action=transition_actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, n_agents),
            discount=jnp.repeat(1 - nstate.done, n_agents),
            extras={"state_extras": state_extras},
        )
    
    def deterministic_actor_step(learner_state, env, env_state, extra_fields):
        """Evaluation step - deterministic policy."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        means, _ = actor.apply(learner_state.params.actor, obs)
        
        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        
        if not discrete_actions:
            means = means - ((1 - avail_actions) * 1e10)
            actions = nn.tanh(means)
        else:
            actions = jnp.argmax(means, axis=-1)
        
        actions_ = jnp.reshape(actions, (-1, n_agents) + actions.shape[1:])
        nstate = env.step(env_state, actions_)
        
        state_extras = {x: jnp.repeat(nstate.info[x], n_agents) for x in extra_fields}
        
        return nstate, Transition(
            observation=obs,
            action=actions if discrete_actions else actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, n_agents),
            discount=jnp.repeat(1 - nstate.done, n_agents),
            extras={"state_extras": state_extras},
        )
    
    @jax.jit
    def get_experience(actor_params, env_state, buffer_state, key):
        """Collect experience and add to buffer."""
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(
                actor_params, env, env_state, current_key, 
                extra_fields=("truncation", "seed")
            )
            return (env_state, next_key), transition
        
        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=rollout_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state
    
    @jax.jit
    def update_critic(transitions, learner_state, key):
        """Update critic using contrastive loss."""
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            obs = transitions.observation[:, :obs_dim]
            action = transitions.action
            
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, obs_dim:])
            
            # InfoNCE loss
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
            
            # Logsumexp regularization
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)
            
            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
            
            return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)
        
        critic_params = {
            "sa_encoder": learner_state.params.sa_encoder,
            "g_encoder": learner_state.params.goal_encoder,
        }
        
        (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(critic_params, transitions, key)
        
        new_critic_state = learner_state.opt_states.critic
        updates, new_critic_state = critic_update_fn(grad, new_critic_state)
        
        new_sa_encoder = optax.apply_updates(learner_state.params.sa_encoder, updates["sa_encoder"])
        new_g_encoder = optax.apply_updates(learner_state.params.goal_encoder, updates["g_encoder"])
        
        new_params = SACCRLParams(
            actor=learner_state.params.actor,
            sa_encoder=new_sa_encoder,
            goal_encoder=new_g_encoder,
        )
        new_opt_states = SACOptStates(
            actor=learner_state.opt_states.actor,
            critic=new_critic_state,
            alpha=learner_state.opt_states.alpha,
        )
        
        learner_state = learner_state._replace(params=new_params, opt_states=new_opt_states)
        
        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }
        
        return learner_state, metrics
    
    @jax.jit
    def update_actor_and_alpha(transitions, learner_state, key):
        """Update actor and alpha."""
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, :obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, goal_start_idx : goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)
            avail_actions = transitions.avail_actions
            
            means, log_stds = actor.apply(actor_params, observation)
            
            if not discrete_actions:
                stds = jnp.exp(log_stds)
                means = means - ((1 - avail_actions) * 1e10)
                x_ts = means + stds * jax.random.gumbel(key, shape=means.shape, dtype=means.dtype)
                action = nn.tanh(x_ts)
                
                # Gumbel log probability
                x_std = (x_ts - means) / stds
                log_prob = -(x_std + jnp.exp(-x_std)) - log_stds
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = jnp.where(avail_actions == 0, 0, log_prob)
                log_prob = log_prob.sum(-1)
            else:
                pi = distrax.Categorical(logits=means)
                action = pi.sample(seed=key)
                log_prob = pi.log_prob(action)
                action = jax.nn.one_hot(action, means.shape[1])
            
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(sa_encoder_params, state, action)
            g_repr = g_encoder.apply(g_encoder_params, goal)
            
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            
            actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)
            
            return actor_loss, log_prob
        
        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy))
            return jnp.mean(alpha_loss)
        
        critic_params = {
            "sa_encoder": learner_state.params.sa_encoder,
            "g_encoder": learner_state.params.goal_encoder,
        }
        
        (actor_loss_val, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            learner_state.params.actor, critic_params, learner_state.alpha_params['log_alpha'], 
            transitions, key
        )
        
        new_actor_state = learner_state.opt_states.actor
        actor_updates, new_actor_state = actor_update_fn(actor_grad, new_actor_state)
        new_actor_params = optax.apply_updates(learner_state.params.actor, actor_updates)
        
        alpha_loss_val, alpha_grad = jax.value_and_grad(alpha_loss)(learner_state.alpha_params, log_prob)
        
        new_alpha_opt_state = learner_state.opt_states.alpha
        alpha_updates, new_alpha_opt_state = alpha_update_fn(alpha_grad, new_alpha_opt_state)
        new_alpha_params = optax.apply_updates(learner_state.alpha_params, alpha_updates)
        
        new_params = SACCRLParams(
            actor=new_actor_params,
            sa_encoder=learner_state.params.sa_encoder,
            goal_encoder=learner_state.params.goal_encoder,
        )
        new_opt_states = SACOptStates(
            actor=new_actor_state,
            critic=learner_state.opt_states.critic,
            alpha=new_alpha_opt_state,
        )
        
        learner_state = learner_state._replace(
            params=new_params, 
            opt_states=new_opt_states,
            alpha_params=new_alpha_params
        )
        
        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actor_loss_val,
            "alpha_loss": alpha_loss_val,
            "log_alpha": learner_state.alpha_params["log_alpha"],
        }
        
        return learner_state, metrics
    
    @jax.jit
    def sgd_step(carry, transitions):
        """Single SGD step on a mini-batch."""
        learner_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)
        
        # IMPORTANT: Update actor BEFORE critic (as in reference implementation)
        learner_state, actor_metrics = update_actor_and_alpha(transitions, learner_state, actor_key)
        learner_state, critic_metrics = update_critic(transitions, learner_state, critic_key)
        
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (learner_state, key), metrics
    
    @jax.jit
    def training_step(learner_state: SACCRLLearnerState, key):
        """One training step: collect experience, sample from buffer, train."""
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)
        
        # Collect experience
        env_state, buffer_state = get_experience(
            learner_state.params.actor,
            learner_state.env_state,
            learner_state.buffer_state,
            experience_key1,
        )
        
        # Sample from buffer
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        
        # Apply HER relabeling
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_crl_fn, in_axes=(None, 0, 0))(
            buffer_config, transitions, batch_keys
        )
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )
        
        # Train on mini-batches
        (learner_state, _), metrics = jax.lax.scan(sgd_step, (learner_state, training_key), transitions)
        
        # Update learner state
        learner_state = learner_state._replace(
            env_state=env_state,
            buffer_state=buffer_state,
            t=learner_state.t + num_envs * rollout_length,
        )
        
        return learner_state, metrics
    
    def learner_fn(learner_state: SACCRLLearnerState) -> ExperimentOutput:
        """Learner function - performs multiple training steps."""
        @jax.jit
        def f(carry, unused_t):
            learner_state, key = carry
            key, train_key = jax.random.split(key, 2)
            learner_state, metrics = training_step(learner_state, train_key)
            return (learner_state, key), metrics
        
        (learner_state, _), train_metrics = jax.lax.scan(
            f, (learner_state, learner_state.key), (), 
            length=config.system.num_updates_per_eval
        )
        
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics={},
            train_metrics=train_metrics,
        )
    
    return learner_fn, deterministic_actor_step


def learner_setup(config: DictConfig) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states."""
    n_devices = len(jax.devices())
    
    # PRNG keys
    key = jax.random.PRNGKey(config.system.seed)
    key, env_key, eval_env_key, actor_key, sa_key, g_key, buffer_key = jax.random.split(key, 7)
    
    # Environment setup
    env = SmaxEnv(map_name=config.env.scenario.task_name)
    
    # Store dimensions in config
    config.system.obs_dim = env.observation_size - 1
    config.system.goal_start_idx = env.observation_size - 2
    config.system.goal_end_idx = env.observation_size - 1
    config.system.num_agents = env.env.num_agents
    config.system.action_size = env.action_size
    
    # Compute target entropy
    config.system.target_entropy = -0.5 * config.system.action_size
    
    # Wrap with Brax training wrapper
    env = envs.training.wrap(
        env,
        episode_length=config.env.time_limit,
    )
    
    # Initialize environment states
    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    # Create networks
    actor = Actor(action_size=config.system.action_size)
    sa_encoder = SA_encoder(rep_size=config.system.rep_size)
    g_encoder = G_encoder(rep_size=config.system.rep_size)
    
    # Initialize network parameters
    obs_size = env.observation_size
    actor_params = actor.init(actor_key, np.ones([1, obs_size]))
    sa_encoder_params = sa_encoder.init(
        sa_key, jnp.ones([1, config.system.obs_dim]), jnp.ones([1, config.system.action_size])
    )
    g_encoder_params = g_encoder.init(
        g_key, jnp.ones([1, config.system.goal_end_idx - config.system.goal_start_idx])
    )
    
    # Pack parameters
    params = SACCRLParams(
        actor=actor_params,
        sa_encoder=sa_encoder_params,
        goal_encoder=g_encoder_params,
    )
    
    # Alpha parameter
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_params = {"log_alpha": log_alpha}
    
    # Optimizers
    actor_opt = optax.adam(learning_rate=config.system.actor_lr)
    actor_opt_state = actor_opt.init(actor_params)
    
    critic_opt = optax.adam(learning_rate=config.system.q_lr)
    critic_opt_state = critic_opt.init({"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params})
    
    alpha_opt = optax.adam(learning_rate=config.system.alpha_lr)
    alpha_opt_state = alpha_opt.init(alpha_params)
    
    opt_states = SACOptStates(
        actor=actor_opt_state,
        critic=critic_opt_state,
        alpha=alpha_opt_state,
    )
    
    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((config.system.action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        avail_actions=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            }
        },
    )
    
    num_envs_agents = num_envs * config.system.num_agents
    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=config.system.max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=config.system.batch_size,
        num_envs=num_envs_agents,
        episode_length=config.env.time_limit,
    )
    
    # JIT buffer methods
    replay_buffer.insert_internal = jax.jit(replay_buffer.insert_internal)
    replay_buffer.sample_internal = jax.jit(replay_buffer.sample_internal)
    
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)
    
    # Initialize timestep counter
    t0 = jnp.zeros((), dtype=jnp.int32)
    
    # Create initial learner state
    init_learner_state = SACCRLLearnerState(
        params=params,
        opt_states=opt_states,
        alpha_params=alpha_params,
        key=key,
        env_state=env_state,
        buffer_state=buffer_state,
        t=t0,
    )
    
    # Get learner function
    learn, deterministic_actor_step = get_learner_fn(
        env, actor, sa_encoder, g_encoder,
        actor_opt.update, critic_opt.update, alpha_opt.update,
        replay_buffer, config
    )
    
    # Create evaluator
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=config.arch.num_eval_envs,
        episode_length=config.env.time_limit,
        key=eval_env_key,
    )
    
    return learn, actor, sa_encoder, g_encoder, init_learner_state, evaluator, env, replay_buffer




def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "sac_crl_brax"
    config = copy.deepcopy(_config)
    
    # Setup learner and environment
    learn, actor, sa_encoder, g_encoder, learner_state, evaluator, env, replay_buffer = learner_setup(config)
    
    jax.block_until_ready(learner_state)
    
    # Prefill replay buffer
    print(f"Prefilling replay buffer to {config.system.min_replay_size} transitions...")
    num_prefill_steps = int(np.ceil(config.system.min_replay_size / config.system.rollout_length))
    
    @jax.jit
    def prefill_step(carry, unused):
        learner_state, key = carry
        key, exp_key = jax.random.split(key)
        
        # Collect experience without training
        @jax.jit
        def collect_rollout(actor_params, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                
                # Simple actor step
                obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
                means, log_stds = actor.apply(actor_params, obs)
                
                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
                
                if not config.system.discrete_actions:
                    stds = jnp.exp(log_stds)
                    noise = jax.random.gumbel(current_key, shape=means.shape, dtype=means.dtype)
                    means = means - ((1 - avail_actions) * 1e10)
                    actions = nn.tanh(means + stds * noise)
                    transition_actions = actions
                else:
                    action_logits = means - ((1 - avail_actions) * 1e10)
                    pi = distrax.Categorical(logits=action_logits)
                    actions = pi.sample(seed=current_key)
                    transition_actions = jax.nn.one_hot(actions, means.shape[1])
                
                actions_ = jnp.reshape(actions, (-1, config.system.num_agents) + actions.shape[1:])
                nstate = env.step(env_state, actions_)
                
                state_extras = {
                    "truncation": jnp.repeat(nstate.info["truncation"], config.system.num_agents),
                    "seed": jnp.repeat(nstate.info["seed"], config.system.num_agents),
                }
                
                return (nstate, next_key), Transition(
                    observation=obs,
                    action=transition_actions,
                    avail_actions=avail_actions,
                    reward=jnp.repeat(nstate.reward, config.system.num_agents),
                    discount=jnp.repeat(1 - nstate.done, config.system.num_agents),
                    extras={"state_extras": state_extras},
                )
            
            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=config.system.rollout_length)
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state
        
        env_state, buffer_state = collect_rollout(
            learner_state.params.actor,
            learner_state.env_state,
            learner_state.buffer_state,
            exp_key,
        )
        
        learner_state = learner_state._replace(
            env_state=env_state,
            buffer_state=buffer_state,
        )
        return (learner_state, key), None
    
    (learner_state, _), _ = jax.lax.scan(
        prefill_step, (learner_state, learner_state.key), None, 
        length=num_prefill_steps
    )
    print(f"Buffer prefilled with {replay_buffer.size(learner_state.buffer_state)} transitions")
    
    jax.block_until_ready(learner_state)
    
    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates must be greater than number of evaluations."
    
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = config.system.num_updates_per_eval * config.system.rollout_length * config.arch.num_envs
    
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    
    # Checkpointing setup
    save_checkpoint = config.logger.checkpointing.save_model
    checkpoint_dir = None
    if save_checkpoint:
        scenario_name = config.env.scenario.task_name
        seed = config.system.seed
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("checkpoints", f"sac_crl_brax_{scenario_name}_seed{seed}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    max_win_rate = -jnp.inf
    
    # Main training loop
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
        train_metrics["steps_per_second"] = sps
        
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)
        
        # Run evaluation
        eval_metrics = evaluator.run_evaluation(learner_output.learner_state, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        
        # Print key metrics
        critic_loss = float(train_metrics["critic_loss"])
        actor_loss = float(train_metrics["actor_loss"])
        log_alpha = float(train_metrics["log_alpha"])
        win_rate = float(eval_metrics.get("win_rate", 0.0))
        episode_return = float(eval_metrics.get("episode_return", 0.0))

        # Save best checkpoint
        if save_checkpoint and checkpoint_dir:
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                
                current_params = {
                    "actor": jax.tree_util.tree_map(lambda x: np.array(x), learner_output.learner_state.params.actor),
                    "sa_encoder": jax.tree_util.tree_map(lambda x: np.array(x), learner_output.learner_state.params.sa_encoder),
                    "goal_encoder": jax.tree_util.tree_map(lambda x: np.array(x), learner_output.learner_state.params.goal_encoder),
                    "alpha": learner_output.learner_state.alpha_params,
                }
                
                best_path = os.path.join(checkpoint_dir, "best_model.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump({
                        "params": current_params,
                        "timestep": t,
                        "win_rate": float(win_rate),
                        "config": OmegaConf.to_container(config, resolve=True),
                    }, f)
                print(f"  Saved best model (win_rate={win_rate:.2f}%) to {best_path}")
        
        learner_state = learner_output.learner_state
    
    # Save final checkpoint
    if save_checkpoint and checkpoint_dir:
        final_params = {
            "actor": jax.tree_util.tree_map(lambda x: np.array(x), learner_state.params.actor),
            "sa_encoder": jax.tree_util.tree_map(lambda x: np.array(x), learner_state.params.sa_encoder),
            "goal_encoder": jax.tree_util.tree_map(lambda x: np.array(x), learner_state.params.goal_encoder),
            "alpha": learner_state.alpha_params,
        }
        final_path = os.path.join(checkpoint_dir, "final_model.pkl")
        with open(final_path, "wb") as f:
            pickle.dump({
                "params": final_params,
                "timestep": t,
                "win_rate": float(eval_metrics.get("win_rate", 0.0)),
                "config": OmegaConf.to_container(config, resolve=True),
            }, f)
        print(f"Saved final model to {final_path}")
    
    eval_performance = float(eval_metrics.get("win_rate", 0.0))
    
    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="sac_crl_brax.yaml",
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
