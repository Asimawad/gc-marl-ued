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

import chex
import flax
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_icrl_eval_fn
from mava.networks import GoalEncoder
from mava.networks import PQNStateActionEncoder as SAEncoder
from mava.systems.icrl.losses import compute_contrastive_metrics, compute_logits, contrastive_loss_fn
from mava.systems.icrl.types import ICRLParams, OptStates, PQNLearnerState
from mava.systems.icrl.types import Transition as ICRLTransition
from mava.types import ExperimentOutput, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger


def flatten_crl_fn(buffer_config, transition, sample_key):
    gamma, obs_dim, goal_start_idx, goal_end_idx = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
    )

    seed_mask = jnp.equal(single_trajectories, single_trajectories.T)

    probs = probs * seed_mask + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state_sampled = jnp.take(transition.observation, goal_index[:-1], axis=0)
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state_sampled[:, goal_start_idx:goal_end_idx]

    future_state = future_state_sampled[:, :obs_dim]
    state = transition.observation[:-1, :obs_dim]
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
    env: MarlEnv,
    sa_encoder_apply_fn,
    g_encoder_apply_fn,
    critic_update_fn,
    config: DictConfig,
) -> Any:
    """Get the learner function for PQN-CRL (no replay buffer)."""

    # Multi-agent dimensions
    n_agents = env.num_agents
    action_dim = env.action_dim
    num_envs_agents = config.arch.num_envs * n_agents

    base_obs_dim = config.system.icrl.obs_dim
    obs_dim = base_obs_dim + (config.system.icrl.goal_end_idx - config.system.icrl.goal_start_idx)
    goal_start_idx = config.system.icrl.goal_start_idx
    goal_end_idx = config.system.icrl.goal_end_idx

    # Epsilon-greedy exploration
    eps_start = config.system.get("eps_start", 1.0)
    eps_min = config.system.get("eps_min", 0.05)
    eps_decay = config.system.get("eps_decay", 50000)

    # Logsumexp penalty coefficient (for contrastive loss)
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff

    # Contrastive learning config
    energy_fn_name = config.system.get("energy_fn", "norm")
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "fwd_infonce")

    # Number of epochs to train on each rollout (1 = pure PQN, >1 = PPO-style reuse)
    num_epochs = config.system.get("num_epochs", 1)

    def _env_step(learner_state: PQNLearnerState, _: Any) -> Tuple[PQNLearnerState, Tuple[dict, ICRLTransition]]:
        """Step the environment for rollout_length steps and return trajectory."""
        params, opt_states, key, env_state, last_timestep, t = learner_state

        def single_step(carry, _):
            """Single environment step with epsilon-greedy action selection."""
            key, env_state, last_timestep, t = carry

            # RNG
            key, action_key = jax.random.split(key)

            # Get observations
            obs = last_timestep.observation.agents_view  # [N_env, N_agent, obs_dim]
            action_mask = last_timestep.observation.action_mask  # [N_env, N_agent, A]

            # Extract state and goal
            state = obs[..., :obs_dim]  # [N_env, N_agent, state_dim]
            goal = obs[..., obs_dim:]  # [N_env, N_agent, goal_dim]

            # Get per-action state representations
            # Shape: [N_env, N_agent, num_actions, rep_size]
            s_repr = sa_encoder_apply_fn(params.sa_encoder, state)

            # Get goal representation
            # Shape: [N_env, N_agent, rep_size]
            g_repr = g_encoder_apply_fn(params.goal_encoder, goal)

            # Compute Q-values as negative distance
            g_repr_expanded = g_repr[..., None, :]  # [N_env, N_agent, 1, rep_size]
            q_values = -jnp.sqrt(
                jnp.sum((s_repr - g_repr_expanded) ** 2, axis=-1) + 1e-8
            )  # i think this should be consistent with the energy function we use for our updates
            # Shape: [N_env, N_agent, num_actions]

            # Mask invalid actions
            q_values = jnp.where(action_mask, q_values, -1e10)

            # Epsilon-greedy action selection
            greedy_actions = jnp.argmax(q_values, axis=-1)  # [N_env, N_agent]

            # Random actions (uniform over valid actions)
            random_actions = jax.random.categorical(
                action_key, jnp.log(action_mask.astype(jnp.float32) + 1e-8), axis=-1
            )

            # Compute epsilon with linear decay
            eps = jnp.maximum(eps_min, eps_start - t * (eps_start - eps_min) / eps_decay)

            # Choose between greedy and random based on epsilon
            explore_key, action_key = jax.random.split(action_key)
            explore = jax.random.uniform(explore_key, greedy_actions.shape) < eps
            actions = jnp.where(explore, random_actions, greedy_actions)

            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, actions)

            # Create one-hot actions
            actions_onehot = jax.nn.one_hot(actions, action_dim)

            # Flatten for processing
            flat_obs = last_timestep.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = actions_onehot.reshape(num_envs_agents, -1)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = action_mask.reshape(num_envs_agents, -1)

            # Broadcast seed and truncation to all agents
            trunc_per_env = timestep.extras["truncation"]
            seed_per_env = timestep.extras["seed"]
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

            # Increment timestep counter
            return (key, env_state, timestep, t + config.arch.num_envs), transition

        # Collect transitions
        (key, env_state, last_timestep, t), traj_batch = jax.lax.scan(
            single_step, (key, env_state, last_timestep, t), None, config.system.rollout_length
        )

        # Get episode metrics
        metrics = last_timestep.extras.get("episode_metrics", {})

        learner_state = PQNLearnerState(params, opt_states, key, env_state, last_timestep, t)
        return learner_state, (metrics, traj_batch)

    def _update_step(learner_state: PQNLearnerState, _: Any) -> Tuple[PQNLearnerState, Tuple]:
        """A single update: collect rollout and train on it directly (no buffer)."""

        # Collect experience - returns trajectory directly
        learner_state, (episode_metrics, traj_batch) = _env_step(learner_state, None)

        params, opt_states, key, env_state, last_timestep, t = learner_state

        key, sample_key = jax.random.split(key)

        # traj_batch has shape (rollout_length, num_envs_agents, ...)
        # Transpose to (num_envs_agents, rollout_length, ...) for hindsight relabeling
        transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # Apply hindsight relabeling to collected trajectory
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_crl_fn, in_axes=(None, 0, 0))(
            (config.system.gamma, obs_dim, goal_start_idx, goal_end_idx), transitions, batch_keys
        )
        # Output shape: (num_envs_agents, rollout_length-1, ...)

        # Reshape with Fortran order to flatten env-agents and time
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        # Truncate to make evenly divisible by batch_size
        num_samples = (len(transitions.observation) // config.system.batch_size) * config.system.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_samples], transitions)

        # Note: transitions stay flat here; shuffling and batching happen inside each epoch

        def _update_minibatch(carry, batch_transitions):
            """Update critic on a single minibatch."""
            params, opt_states, key = carry
            key, critic_key = jax.random.split(key)

            # Extract obs and action for this batch
            obs = batch_transitions.observation  # [batch_size, obs_dim+goal_dim]
            action_onehot = batch_transitions.action  # [batch_size, action_dim]

            def _critic_loss_fn(critic_params, obs, action_onehot):
                """Pure contrastive loss."""

                # After relabeling: obs = concat([state, goal])
                state = obs[:, :obs_dim]
                goal = obs[:, obs_dim:]

                # Get per-action state representations: [batch, num_actions, rep_size]
                sa_repr_all = sa_encoder_apply_fn(critic_params["sa_encoder"], state)

                # Get representation for the taken action
                action_indices = jnp.argmax(action_onehot, axis=-1)  # [batch]
                sa_repr = sa_repr_all[jnp.arange(sa_repr_all.shape[0]), action_indices, :]  # [batch, rep_size]

                # Get goal representation: [batch, rep_size]
                g_repr = g_encoder_apply_fn(critic_params["goal_encoder"], goal)

                # Compute pairwise logits for InfoNCE
                logits = compute_logits(energy_fn_name, sa_repr, g_repr)  # [batch, batch]

                # Compute contrastive loss
                critic_loss = contrastive_loss_fn(contrastive_loss_name, logits, logsumexp_penalty_coeff)

                # Metrics for monitoring
                metrics = compute_contrastive_metrics(logits)

                # Also compute Q-value metrics for the taken actions
                g_repr_expanded = g_repr[:, None, :]  # [batch, 1, rep_size]
                q_all = -jnp.sqrt(jnp.sum((sa_repr_all - g_repr_expanded) ** 2, axis=-1) + 1e-8)
                q_taken = q_all[jnp.arange(q_all.shape[0]), action_indices]

                loss_info = {
                    "critic_loss": critic_loss,
                    "logits_pos": metrics["logits_pos"],
                    "logits_neg": metrics["logits_neg"],
                    "categorical_accuracy": metrics["categorical_accuracy"],
                    "logsumexp": metrics["logsumexp"],
                    "q_mean": q_taken.mean(),
                }
                return critic_loss, loss_info

            # Pack params for gradient computation
            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}

            # Update critic
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (critic_loss, critic_info), critic_grads = critic_grad_fn(critic_params, obs, action_onehot)

            # Average gradients across devices
            critic_grads, critic_info = lax.pmean((critic_grads, critic_info), axis_name="device")

            # Apply critic updates
            critic_updates, new_critic_opt_state = critic_update_fn(critic_grads, opt_states.critic)
            new_sa_encoder = optax.apply_updates(params.sa_encoder, critic_updates["sa_encoder"])
            new_goal_encoder = optax.apply_updates(params.goal_encoder, critic_updates["goal_encoder"])

            # Package new params and opt_states
            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
            )
            new_opt_states = OptStates(critic=new_critic_opt_state)

            return (new_params, new_opt_states, key), critic_info

        def _train_epoch(carry, _):
            """Train for one epoch over all data (with reshuffling)."""
            params, opt_states, key, flat_transitions = carry

            # Reshuffle data for this epoch
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, flat_transitions.observation.shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_transitions)

            # Reshape into minibatches
            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
                shuffled,
            )

            # Train on all minibatches
            (params, opt_states, key), epoch_metrics = jax.lax.scan(
                _update_minibatch, (params, opt_states, key), batched
            )

            return (params, opt_states, key, flat_transitions), epoch_metrics

        # Train for num_epochs (each epoch reshuffles and trains on all data)
        (params, opt_states, key, _), train_metrics = jax.lax.scan(
            _train_epoch, (params, opt_states, key, transitions), None, num_epochs
        )

        learner_state = PQNLearnerState(params, opt_states, key, env_state, last_timestep, t)
        return learner_state, (episode_metrics, train_metrics)

    def learner_fn(learner_state: PQNLearnerState) -> ExperimentOutput:
        """Learner function - performs multiple update steps."""
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


def learner_setup(env: MarlEnv, keys: chex.Array, config: DictConfig) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states (no buffer)."""
    n_devices = len(jax.devices())
    config.system.num_agents = env.num_agents

    # PRNG keys
    key, sa_key, g_key = keys

    # Define networks with configurable torsos
    sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)

    # Create encoders
    sa_encoder = SAEncoder(torso=sa_encoder_torso, num_actions=env.action_dim, output_dim=64)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, output_dim=64)

    # Initialize network parameters
    base_obs_dim = config.system.icrl.obs_dim
    obs_dim = base_obs_dim + (config.system.icrl.goal_end_idx - config.system.icrl.goal_start_idx)

    # Create dummy inputs for initialization
    init_state = jnp.zeros((1, obs_dim))
    init_goal = jnp.zeros((1, (config.system.icrl.goal_end_idx - config.system.icrl.goal_start_idx)))

    # Initialize encoders
    sa_encoder_params = sa_encoder.init(sa_key, init_state)
    goal_encoder_params = goal_encoder.init(g_key, init_goal)

    # Pack parameters
    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
    )

    # Calculate total gradient steps for LR scheduler
    num_epochs = config.system.get("num_epochs", 1)
    # Approximate number of minibatches per update (rough estimate)
    samples_per_update = config.arch.num_envs * env.num_agents * (config.system.rollout_length - 1)
    minibatches_per_update = samples_per_update // config.system.batch_size
    total_grad_steps = config.system.num_updates * num_epochs * minibatches_per_update

    # LR scheduler (optional - controlled by lr_linear_decay config)
    lr_linear_decay = config.system.get("lr_linear_decay", False)
    lr_end = config.system.get("lr_end", 1e-7)

    if lr_linear_decay:
        lr_scheduler = optax.linear_schedule(
            init_value=config.system.q_lr,
            end_value=lr_end,
            transition_steps=total_grad_steps,
        )
        lr = lr_scheduler
    else:
        lr = config.system.q_lr

    # Optimizer - single optimizer for both encoders
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    critic_opt = optax.chain(grad_clip, optax.radam(lr))
    critic_opt_state = critic_opt.init({"sa_encoder": sa_encoder_params, "goal_encoder": goal_encoder_params})

    opt_states = OptStates(critic=critic_opt_state)

    # Get learner function and pmap it
    learn = get_learner_fn(env, sa_encoder.apply, goal_encoder.apply, critic_opt.update, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialize environment states
    key, *env_keys = jax.random.split(key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1)
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(jnp.stack(env_keys))

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

    # Initialize timestep counter for epsilon decay
    t0 = jnp.zeros((n_devices, config.system.update_batch_size), dtype=jnp.int32)

    # Replicate params, opt_states
    params_broadcast = tree.map(lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape)), params)
    opt_states_broadcast = tree.map(
        lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape)), opt_states
    )

    replicate_items = (params_broadcast, opt_states_broadcast)
    replicate_items = flax.jax_utils.replicate(replicate_items, devices=jax.devices())

    params, opt_states = replicate_items
    init_learner_state = PQNLearnerState(params, opt_states, step_keys, env_states, timesteps, t0)

    return learn, sa_encoder, goal_encoder, init_learner_state


def make_crl_eval_act_fn(sa_encoder_apply_fn, g_encoder_apply_fn, config):
    """Create evaluation action function for PQN-CRL."""
    obs_dim = config.system.icrl.obs_dim + config.system.icrl.goal_end_idx - config.system.icrl.goal_start_idx

    def eval_act_fn(params, timestep, key, actor_state):
        """Select greedy action based on Q-values (argmax, no exploration)."""
        observation = timestep.observation
        obs = observation.agents_view
        action_mask = observation.action_mask

        # Extract state and goal
        state = obs[..., :obs_dim]
        goal = obs[..., obs_dim:]

        # Get per-action state representations
        s_repr = sa_encoder_apply_fn(params["sa_encoder"], state)

        # Get goal representation
        g_repr = g_encoder_apply_fn(params["goal_encoder"], goal)

        # Compute Q-values as negative distance
        g_repr_expanded = g_repr[..., None, :]
        q_values = -jnp.sqrt(jnp.sum((s_repr - g_repr_expanded) ** 2, axis=-1) + 1e-8)

        # Mask invalid actions
        q_values = jnp.where(action_mask, q_values, -1e10)

        # Greedy action (no exploration during eval)
        actions = jnp.argmax(q_values, axis=-1)

        return actions, actor_state

    return eval_act_fn


def run_experiment(_config: DictConfig) -> float:
    """Run experiment."""
    _config.logger.system_name = "pqn_crl"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create environments
    env, eval_env = environments.make(config)

    # PRNG keys
    key, key_e, sa_key, g_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=4)

    # No buffer key needed for PQN
    learn, sa_encoder, goal_encoder, learner_state = learner_setup(env, (key, sa_key, g_key), config)

    jax.block_until_ready(learner_state)

    # Setup evaluator
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_crl_eval_act_fn(sa_encoder.apply, goal_encoder.apply, config)
    evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates must be greater than number of evaluations."

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

    # Checkpointer
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
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        episode_metrics = learner_output.episode_metrics
        if episode_metrics:
            episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if episode_metrics:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluation
        trained_params = {
            "sa_encoder": unreplicate_batch_dim(learner_state.params.sa_encoder),
            "goal_encoder": unreplicate_batch_dim(learner_state.params.goal_encoder),
        }
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)

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

        learner_state = learner_output.learner_state

    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    if config.arch.absolute_metric:
        abs_metric_evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="pqn_crl.yaml",
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
        except:
            override_info = "unknown"
        print(f"Error executing job with overrides: {override_info}")
        print(f"Exception: {type(e).__name__}: {e!s}")
        import traceback

        traceback.print_exc()
        return float("-inf")


if __name__ == "__main__":
    hydra_entry_point()
