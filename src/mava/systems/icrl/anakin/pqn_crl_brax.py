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
from brax import envs
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
from hydra.utils import instantiate
from jax import tree
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

# Working infrastructure imports dont touch them
from mava.evaluator import CrlEvaluator
from mava.wrappers.smax import SmaxEnv

# Mava imports
from mava.systems.icrl.types import Transition, ICRLParams, OptStates
from mava.types import ExperimentOutput
# Note: Using simple pickle-based checkpointing instead of orbax (version compatibility issues)
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger

import flax.linen as nn

def flatten_crl_fn(buffer_config, transition, sample_key):

    gamma, obs_dim, goal_start_idx, goal_end_idx = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32) # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)        
    probs = is_future_mask * discount  

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))
    
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
    )
    # array of seq_len x seq_len where a row is an array of seeds that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same seed. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    #ith row of probs will be non zero only for time indices that 
    # 1) are greater than i
    # 2) have the same seed as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(transition.observation, goal_index[:-1], axis=0) #the last goal_index cannot be considered as there is no future.  
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_start_idx : goal_end_idx]
    future_state = future_state[:, : obs_dim]
    state = transition.observation[:-1, : obs_dim] #all states are considered
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
        observation=jnp.squeeze(new_obs),   #this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )

class SA_encoder(nn.Module):
    """
    State-Action encoder with explicit action encoding.
    LayerNorm is crucial for PQN stability!
    """

    rep_size: int
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # LayerNorm is key for PQN stability
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

    rep_size: int
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


class BraxPQNLearnerState(NamedTuple):
    """Learner state for PQN-CRL with Brax environments."""

    params: ICRLParams
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: Any  # Brax State
    t: chex.Array  # Timestep counter
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
) -> Any:
    """Get the learner function for PQN-CRL with Brax infrastructure."""

    # Extract config values
    n_agents = config.system.num_agents
    action_size = config.system.action_size
    num_envs = config.arch.num_envs
    num_envs_agents = num_envs * n_agents
    rollout_length = config.system.rollout_length
    batch_size = config.system.batch_size
    gamma = config.system.gamma
    temperature = config.system.temperature
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    use_target_network = config.system.use_target_network
    target_tau = config.system.target_tau

    # Epsilon-greedy exploration
    eps_start = config.system.get("eps_start", 1.0)
    eps_min = config.system.get("eps_min", 0.05)
    eps_decay = config.system.get("eps_decay", 15000)
    # def get_epsilon(t):
    #     return max(eps_min, eps_start - (eps_start - eps_min) * (t / eps_decay))
    # Observation dimensions
    obs_dim = config.system.obs_dim
    goal_start_idx = config.system.goal_start_idx
    goal_end_idx = config.system.goal_end_idx

    # Buffer config tuple for flatten_crl_fn
    buffer_config = (gamma, obs_dim, goal_start_idx, goal_end_idx)

    # Get available actions function
    get_avail_act = jax.vmap(env.get_avail_actions)

    def compute_q_values(critic_params, obs, goal):
        """Compute Q-values for all actions."""
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]

        batch_size_local = obs.shape[0]
        g_repr = g_encoder.apply(g_encoder_params, goal)

        def compute_q_for_action(action_idx):
            a_onehot = jax.nn.one_hot(jnp.full(batch_size_local, action_idx), action_size)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, a_onehot)
            q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            return q

        q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_size))
        q_values = q_values.T

        return q_values

    def get_action_params(learner_state: BraxPQNLearnerState):
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

    def actor_step(action_params, env_state, key, t, extra_fields):
        """Training step - epsilon-greedy action selection."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])

        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]

        logits = compute_q_values(action_params, state, goal)

        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        q_values = logits - ((1 - avail_actions) * 1e10)

        # Epsilon-greedy action selection
        greedy_actions = jnp.argmax(q_values, axis=-1)

        # Split key for random actions
        random_key, explore_key = jax.random.split(key, 2)

        # Random actions (uniform over valid actions)
        random_actions = jax.random.categorical(
            random_key, jnp.log(avail_actions.astype(jnp.float32) + 1e-8), axis=-1
        )

        # Compute epsilon with linear decay
        eps = jnp.maximum(eps_min, eps_start - t * (eps_start - eps_min) / eps_decay)
        # eps = get_epsilon(t)

        # Choose between greedy and random based on epsilon
        explore = jax.random.uniform(explore_key, greedy_actions.shape) < eps
        actions = jnp.where(explore, random_actions, greedy_actions)

        actions_ = jnp.reshape(actions, (-1, n_agents))
        nstate = env.step(env_state, actions_)

        state_extras = {x: jnp.repeat(nstate.info[x], n_agents) for x in extra_fields}

        return nstate, Transition(
            observation=obs,
            action=actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, n_agents),
            discount=jnp.repeat(1 - nstate.done, n_agents),
            extras={"state_extras": state_extras},
        )

    def deterministic_actor_step(learner_state, env, env_state, extra_fields):
        """Evaluation step - categorical sampling with temperature."""
        temp = config.system.temperature  
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])

        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]

        logits = compute_q_values(get_action_params(learner_state), state, goal)

        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        logits = logits - ((1 - avail_actions) * 1e10)

        # Split key from learner_state for stochastic sampling
        # action_key, _ = jax.random.split(learner_state.key)
        # actions = jax.random.categorical(action_key, logits / temp, axis=-1)
        actions = jnp.argmax(logits, axis=-1)  # Deterministic action selection for eval
        actions_ = jnp.reshape(actions, (-1, n_agents))
        nstate = env.step(env_state, actions_)

        state_extras = {x: jnp.repeat(nstate.info[x], n_agents) for x in extra_fields}

        return nstate, Transition(
            observation=obs,
            action=actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, n_agents),
            discount=jnp.repeat(1 - nstate.done, n_agents),
            extras={"state_extras": state_extras},
        )

    def collect_experience(action_params, env_state, key, t):
        """Collect experience from parallel envs - NO BUFFER, return directly."""

        def f(carry, unused_t):
            env_state, current_key, current_t = carry
            current_key, next_key = jax.random.split(current_key)

            env_state, transition = actor_step(
                action_params,
                env_state,
                current_key,
                current_t,
                extra_fields=("truncation", "seed"),
            )
            # Increment timestep by number of environments
            next_t = current_t + num_envs
            return (env_state, next_key, next_t), transition

        (env_state, _, final_t), transitions = jax.lax.scan(f, (env_state, key, t), (), length=rollout_length)
        return env_state, transitions, final_t

    def update_critic_pqn(transitions, learner_state, key):
        """Update critic on fresh experience (no replay buffer)."""

        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]

            obs = transitions.observation[:, :obs_dim]
            action = transitions.action
            goal = transitions.observation[:, obs_dim:]

            action_onehot = jax.nn.one_hot(action, action_size)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action_onehot)
            g_repr = g_encoder.apply(g_encoder_params, goal)

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

            # Compute Q-values for all actions to measure policy entropy
            def compute_q_for_action(action_idx):
                a_onehot = jax.nn.one_hot(jnp.full(obs.shape[0], action_idx), action_size)
                sa_repr_a = sa_encoder.apply(sa_encoder_params, obs, a_onehot)
                q = -jnp.sqrt(jnp.sum((sa_repr_a - g_repr) ** 2, axis=-1))
                return q

            q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_size))
            q_values = q_values.T  # [batch_size, action_size]
            
            # Compute softmax policy with temperature
            policy = jax.nn.softmax(q_values / temperature, axis=-1)
            
            # Compute entropy: -sum(p * log(p))
            log_policy = jnp.log(policy + 1e-8)
            policy_entropy = -jnp.sum(policy * log_policy, axis=-1)  # [batch_size]
            mean_entropy = jnp.mean(policy_entropy)
            
            # Max possible entropy for reference
            max_entropy = jnp.log(action_size)
            entropy_ratio = mean_entropy / max_entropy  # 0 = deterministic, 1 = uniform

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
                    "goal_encoder": learner_state.target_params.goal_encoder,
                },
                {"sa_encoder": new_sa_encoder, "goal_encoder": new_g_encoder},
                target_tau,
            )
            new_target_params = ICRLParams(
                sa_encoder=new_target_params["sa_encoder"],
                goal_encoder=new_target_params["goal_encoder"],
            )
        else:
            new_target_params = learner_state.target_params

        learner_state = BraxPQNLearnerState(
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
            "entropy_ratio": entropy_ratio,  # 0 = deterministic, 1 = uniform random
            # "epsilon": get_epsilon(learner_state.t),  # placeholder, will be filled later
        }

        return learner_state, metrics

    def sgd_step(carry, transitions):
        """Single SGD step on a mini-batch."""
        learner_state, key = carry
        key, critic_key = jax.random.split(key, 2)
        learner_state, metrics = update_critic_pqn(transitions, learner_state, critic_key)
        return (learner_state, key), metrics

    def _update_step(learner_state: BraxPQNLearnerState, _: Any) -> Tuple[BraxPQNLearnerState, Tuple]:
        """One training step: collect experience and train immediately."""
        key = learner_state.key
        experience_key, train_key, sampling_key, permute_key, new_key = jax.random.split(key, 5)

        # Collect fresh experience (NO BUFFER!)
        env_state, transitions, final_t = collect_experience(
            get_action_params(learner_state),
            learner_state.env_state,
            experience_key,
            learner_state.t,
        )

        # transitions has shape (rollout_length, num_envs_agents, ...)
        # We need to transpose to (num_envs_agents, rollout_length, ...) for vmap
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1),
            transitions,
        )

        # Apply hindsight relabeling using working script's flatten_crl_fn
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

        # Update learner state with new env_state and timestep
        learner_state = BraxPQNLearnerState(
            params=learner_state.params,
            opt_states=learner_state.opt_states,
            key=new_key,
            env_state=env_state,
            t=final_t,
            target_params=learner_state.target_params,
        )

        # Train on all mini-batches
        (learner_state, _), metrics = jax.lax.scan(sgd_step, (learner_state, train_key), transitions)

        # Compute current epsilon for logging
        current_eps = jnp.maximum(eps_min, eps_start - final_t * (eps_start - eps_min) / eps_decay)

        # Add epsilon to metrics (same shape as other metrics for proper averaging)
        eps_metric = jnp.full_like(metrics["critic_loss"], current_eps)

        return learner_state, ({}, {**metrics, "epsilon": eps_metric})

    def learner_fn(learner_state: BraxPQNLearnerState) -> ExperimentOutput:
        """Learner function - performs multiple update steps."""
        learner_state, (episode_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None, config.system.num_updates_per_eval
        )

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_metrics,
            train_metrics=train_metrics,
        )

    return learner_fn, deterministic_actor_step


def learner_setup(config: DictConfig) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states."""
    n_devices = len(jax.devices())

    # PRNG keys
    key = jax.random.PRNGKey(config.system.seed)
    key, env_key, eval_env_key, sa_key, g_key = jax.random.split(key, 5)

    # Environment setup 
    env = SmaxEnv(map_name=config.env.scenario.task_name)

    # Store dimensions in config
    config.system.obs_dim = env.observation_size - 1
    config.system.goal_start_idx = env.observation_size - 2
    config.system.goal_end_idx = env.observation_size - 1
    config.system.num_agents = env.env.num_agents
    config.system.action_size = env.action_size

    # Wrap with Brax training wrapper
    env = envs.training.wrap(
        env,
        episode_length=config.env.time_limit,
    )

    # Initialize environment states
    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, n_devices * num_envs)
    env_keys = env_keys.reshape(n_devices, num_envs, -1)

    # Create env states for each device
    def init_envs(keys):
        return jax.jit(env.reset)(keys)

    env_states = jax.vmap(init_envs)(env_keys)

    # JIT the step function
    env.step = jax.jit(env.step)

    # Create networks 
    sa_encoder = SA_encoder(rep_size=config.system.rep_size)
    g_encoder = G_encoder(rep_size=config.system.rep_size)

    # Initialize network parameters
    sa_encoder_params = sa_encoder.init(
        sa_key, jnp.ones([1, config.system.obs_dim]), jnp.ones([1, config.system.action_size])
    )
    g_encoder_params = g_encoder.init(
        g_key, jnp.ones([1, config.system.goal_end_idx - config.system.goal_start_idx])
    )

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
    learn, deterministic_actor_step = get_learner_fn(env, sa_encoder, g_encoder, critic_opt.update, config)

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
    init_learner_state = BraxPQNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        t=t0,
        target_params=target_params,
    )

    # Create evaluator (using working infrastructure)
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=config.arch.num_eval_envs,
        episode_length=config.env.time_limit,
        key=eval_env_key,
    )

    return learn, sa_encoder, g_encoder, init_learner_state, evaluator, env



def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "pqn_crl_brax"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Setup learner and environment
    learn, sa_encoder, g_encoder, learner_state, evaluator, env = learner_setup(config)

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

    # Simple pickle-based checkpointing (avoids orbax version issues)
    save_checkpoint = config.logger.checkpointing.save_model
    checkpoint_dir = None
    if save_checkpoint:
        # Create checkpoint directory
        scenario_name = config.env.scenario.task_name
        seed = config.system.seed
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("checkpoints", f"pqn_crl_brax_{scenario_name}_seed{seed}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    max_episode_return = -jnp.inf
    max_win_rate = -jnp.inf
    best_params = None



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

        # Log training metrics (no prefix - LogEvent.TRAIN adds "trainer/" automatically)
        train_metrics["steps_per_second"] = sps

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Run evaluation using working evaluator
        # Need to extract unreplicated params for eval
        eval_learner_state = jax.tree_util.tree_map(lambda x: x[0], learner_output.learner_state)
        eval_metrics = evaluator.run_evaluation(eval_learner_state, {})

        # Log evaluation metrics
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        # Print key metrics to console
        epsilon = float(train_metrics["epsilon"])
        critic_loss = float(train_metrics["critic_loss"])
        win_rate = float(eval_metrics.get("win_rate", 0.0))
        episode_return = float(eval_metrics.get("episode_return", 0.0))

        print(
            f"Eval {eval_step+1}/{config.arch.num_evaluation} | "
            f"Step {t} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Loss: {critic_loss:.4f} | "
            f"Win Rate: {win_rate:.2f}% | "
            f"Return: {episode_return:.2f} | "
            f"SPS: {sps:.0f}"
        )

        if save_checkpoint and checkpoint_dir:
            # Get win rate for this evaluation
            win_rate = eval_metrics.get("win_rate", 0.0)
            
            # Save best model based on win rate
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                
                # Extract params (convert to numpy for pickle)
                current_params = {
                    "sa_encoder": jax.tree_util.tree_map(
                        lambda x: np.array(x[0]), learner_output.learner_state.params.sa_encoder
                    ),
                    "goal_encoder": jax.tree_util.tree_map(
                        lambda x: np.array(x[0]), learner_output.learner_state.params.goal_encoder
                    ),
                }
                
                # Save best checkpoint
                best_path = os.path.join(checkpoint_dir, "best_model.pkl")
                with open(best_path, "wb") as f:
                    pickle.dump({
                        "params": current_params,
                        "timestep": t,
                        "win_rate": float(win_rate),
                        "config": OmegaConf.to_container(config, resolve=True),
                    }, f)
                print(f"  Saved best model (win_rate={win_rate:.2f}%) to {best_path}")
            
            # Also save periodic checkpoint every 10 evals
            if eval_step % 10 == 0:
                current_params = {
                    "sa_encoder": jax.tree_util.tree_map(
                        lambda x: np.array(x[0]), learner_output.learner_state.params.sa_encoder
                    ),
                    "goal_encoder": jax.tree_util.tree_map(
                        lambda x: np.array(x[0]), learner_output.learner_state.params.goal_encoder
                    ),
                }
                periodic_path = os.path.join(checkpoint_dir, f"checkpoint_step{t}.pkl")
                with open(periodic_path, "wb") as f:
                    pickle.dump({
                        "params": current_params,
                        "timestep": t,
                        "win_rate": float(win_rate),
                        "config": OmegaConf.to_container(config, resolve=True),
                    }, f)

        learner_state = learner_output.learner_state

    # Save final checkpoint
    if save_checkpoint and checkpoint_dir:
        final_params = {
            "sa_encoder": jax.tree_util.tree_map(
                lambda x: np.array(x[0]), learner_state.params.sa_encoder
            ),
            "goal_encoder": jax.tree_util.tree_map(
                lambda x: np.array(x[0]), learner_state.params.goal_encoder
            ),
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

    # Return final win rate as performance metric
    eval_performance = float(eval_metrics.get("win_rate", 0.0))

    logger.stop()
    return eval_performance




@hydra.main(
    config_path="../../../configs/default",
    config_name="pqn_crl_brax.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes
    OmegaConf.set_struct(cfg, False)

    try:
        # Run experiment
        eval_performance = run_experiment(cfg)
        return eval_performance
    except Exception as e:
        # Log the error but don't crash the sweep
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

