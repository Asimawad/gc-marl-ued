"""
PQN (Parallelized Q-Network) for SMAX - NO REPLAY BUFFER

Key idea: Instead of storing experiences in a buffer and sampling,
we train directly on the freshly collected experiences from parallel envs.
This relies on:
1. Many parallel environments for diverse experience
2. Layer normalization for training stability
3. Online learning without experience replay

Reference: "Simplifying Deep Temporal Difference Learning" (Gallici et al.)
"""

import os
import jax
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from functools import partial

from brax import envs
from etils import epath
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue


@dataclass
class Args:
    exp_name: str = "pqn_smax"
    seed: int = 69
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "CRL_SMAX"
    wandb_entity: str = 'asim_awad'
    wandb_mode: str = 'offline'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    capture_video: bool = False
    checkpoint: bool = False
    load_path: str = ''

    # Environment specific arguments
    env_id: str = "smax"
    smax_map_name: str = "2s3z"
    episode_length: int = 101
    
    # To be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0
    num_agents: int = 0
    action_size: int = 0

    # Algorithm specific arguments
    total_env_steps: int = 50_000_000
    num_epochs: int = 500
    num_envs: int = 512  # More parallel envs for diversity (was 256)
    num_eval_envs: int = 256
    critic_lr: float = 1e-4  # Lower LR for stability (was 3e-4)
    batch_size: int = 256  # Mini-batch size for training (InfoNCE needs small batches)
    rep_size: int = 64
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    max_grad_norm: float = 1.0  # Gradient clipping for stability
    
    # Target network for stability (optional for PQN, but can help)
    use_target_network: bool = True
    target_tau: float = 0.001  # Slower target updates for stability (was 0.005)

    # Temperature for exploration
    temperature: float = 0.05  # Fixed temperature (like your working config)

    # PQN specific: how many env steps to collect before each training update
    unroll_length: int = 100  # Shorter unrolls, more frequent updates
    
    # Number of training epochs per collected batch
    num_updates_per_batch: int = 1  # Usually 1 for PQN (on-policy-ish)

    # To be filled in runtime
    env_steps_per_actor_step: int = 0
    num_training_steps_per_epoch: int = 0
    num_envs_agents: int = 0


class SA_encoder(nn.Module):
    """
    State-Action encoder with explicit action encoding.
    LayerNorm is crucial for PQN stability!
    """
    rep_size: int
    norm_type: str = "layer_norm"  # Important for PQN!
    
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
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
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
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


@flax.struct.dataclass
class TrainingState:
    """Training state for PQN."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    critic_state: TrainState
    target_critic_params: Any  # Optional, can be None if not using target


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    avail_actions: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))


def soft_update(target_params, online_params, tau: float):
    return jax.tree_util.tree_map(
        lambda t, o: (1.0 - tau) * t + tau * o,
        target_params,
        online_params,
    )


if __name__ == "__main__":

    args = tyro.cli(Args)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    # For PQN: simpler calculation since no buffer prefill
    args.num_training_steps_per_epoch = args.total_env_steps // (args.num_epochs * args.env_steps_per_actor_step)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        if args.wandb_group == '.':
            args.wandb_group = None
            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        
    if args.checkpoint:
        from pathlib import Path
        save_path = Path(args.wandb_dir) / Path(run_name)
        os.mkdir(path=save_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, env_key, eval_env_key, sa_key, g_key = jax.random.split(key, 5)

    # Environment setup
    if args.env_id == "smax":
        from envs.smax import SmaxEnv
        env = SmaxEnv(map_name=args.smax_map_name)
        
        args.obs_dim = env.observation_size - 1
        args.goal_start_idx = env.observation_size - 2
        args.goal_end_idx = env.observation_size - 1

        args.num_agents = env.env.num_agents
        args.num_envs_agents = args.num_envs * args.num_agents
        args.action_size = env.action_size
    else:
        raise NotImplementedError(f"Environment {args.env_id} not supported.")
    
    get_avail_act = jax.vmap(env.get_avail_actions)

    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    crtc_params = load_params(args.load_path) if args.load_path else None

    # Network setup
    sa_encoder = SA_encoder(rep_size=args.rep_size)
    sa_encoder_params = sa_encoder.init(
        sa_key, 
        np.ones([1, args.obs_dim]),
        np.ones([1, action_size])
    )
    
    g_encoder = G_encoder(rep_size=args.rep_size)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    
    if crtc_params is None:
        crtc_params = {"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}
    
    critic_state = TrainState.create(
        apply_fn=None,
        params=crtc_params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),  # Gradient clipping
            optax.adam(learning_rate=args.critic_lr),
        ),
    )

    print(f"PQN Mode: No replay buffer, training on fresh experience")
    print(f"Using {args.num_envs} parallel environments")
    print(f"Unroll length: {args.unroll_length}")
    print(f"Batch size per update: {args.num_envs_agents * args.unroll_length}")
    print(f"Target network: {args.use_target_network}")
    
    # Training state
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        critic_state=critic_state,
        target_critic_params=critic_state.params if args.use_target_network else None,
    )

    def compute_q_values(critic_params, obs, goal):
        """Compute Q-values for all actions."""
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]
        
        batch_size = obs.shape[0]
        g_repr = g_encoder.apply(g_encoder_params, goal)
        
        def compute_q_for_action(action_idx):
            a_onehot = jax.nn.one_hot(
                jnp.full(batch_size, action_idx), 
                action_size
            )
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, a_onehot)
            q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
            return q
        
        q_values = jax.vmap(compute_q_for_action)(jnp.arange(action_size))
        q_values = q_values.T
        
        return q_values

    def get_action_params(training_state):
        """Get params to use for action selection."""
        if args.use_target_network:
            return training_state.target_critic_params
        else:
            return training_state.critic_state.params

    def deterministic_actor_step(training_state, env, env_state, extra_fields, key=None):
        """Evaluation step - greedy action selection."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        
        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]
        
        logits = compute_q_values(get_action_params(training_state), state, goal)
        
        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        logits = logits - ((1 - avail_actions) * 1e10)
        
        # actions = jnp.argmax(logits, axis=-1)
        actions= jax.random.categorical(key, logits / args.temperature, axis=-1)
        actions_ = jnp.reshape(actions, (-1, args.num_agents))
        nstate = env.step(env_state, actions_)
        
        state_extras = {x: jnp.repeat(nstate.info[x], args.num_agents) for x in extra_fields}
        
        return nstate, Transition(
            observation=obs,
            action=actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, args.num_agents),
            discount=jnp.repeat(1 - nstate.done, args.num_agents),
            extras={"state_extras": state_extras},
        )

    def actor_step(action_params, env, env_state, key, extra_fields):
        """Training step - sample actions with fixed temperature."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])

        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]

        logits = compute_q_values(action_params, state, goal)

        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        masked_logits = logits - ((1 - avail_actions) * 1e10)

        # Fixed temperature
        actions = jax.random.categorical(key, masked_logits / args.temperature, axis=-1)

        actions_ = jnp.reshape(actions, (-1, args.num_agents))
        nstate = env.step(env_state, actions_)

        state_extras = {x: jnp.repeat(nstate.info[x], args.num_agents) for x in extra_fields}

        return nstate, Transition(
            observation=obs,
            action=actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, args.num_agents),
            discount=jnp.repeat(1 - nstate.done, args.num_agents),
            extras={"state_extras": state_extras},
        )

    @jax.jit
    def collect_experience(action_params, env_state, key):
        """Collect experience from parallel envs - NO BUFFER, return directly."""
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)

            env_state, transition = actor_step(
                action_params,
                env,
                env_state,
                current_key,
                extra_fields=("truncation", "seed"),
            )
            return (env_state, next_key), transition

        (env_state, _), transitions = jax.lax.scan(
            f, (env_state, key), (), length=args.unroll_length
        )
        return env_state, transitions

    # No custom process function - we use TrajectoryUniformSamplingQueue.flatten_crl_fn

    @jax.jit
    def update_critic_pqn(transitions, training_state, key):
        """Update critic on fresh experience (no replay buffer)."""
        
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            obs = transitions.observation[:, :args.obs_dim]
            action = transitions.action
            goal = transitions.observation[:, args.obs_dim:]
            
            action_onehot = jax.nn.one_hot(action, action_size)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action_onehot)
            g_repr = g_encoder.apply(g_encoder_params, goal)
            
            # InfoNCE loss
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # Logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss, (logsumexp, correct, logits_pos, logits_neg)
            
        (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params, transitions, key
        )
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)

        # Update target network if using
        if args.use_target_network:
            new_target_params = soft_update(
                training_state.target_critic_params,
                new_critic_state.params,
                args.target_tau,
            )
        else:
            new_target_params = None

        training_state = training_state.replace(
            critic_state=new_critic_state,
            target_critic_params=new_target_params,
        )

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
        }

        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        """Single SGD step on a mini-batch."""
        training_state, key = carry
        key, critic_key = jax.random.split(key, 2)
        training_state, metrics = update_critic_pqn(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
        return (training_state, key), metrics

    @jax.jit
    def training_step(training_state, env_state, key):
        """One training step: collect experience and train immediately."""
        experience_key, train_key, sampling_key, permute_key = jax.random.split(key, 4)
        
        # Collect fresh experience (NO BUFFER!)
        env_state, transitions = collect_experience(
            get_action_params(training_state),
            env_state,
            experience_key,
        )
        
        # transitions has shape (unroll_length, num_envs_agents, ...)
        # We need to transpose to (num_envs_agents, unroll_length, ...) for vmap
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1),
            transitions,
        )
        
        # Now use flatten_crl_fn just like in v2 code
        # transitions.observation now has shape (num_envs_agents, unroll_length, obs_size)
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        
        # Flatten the batch dimension
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        
        # Shuffle
        permutation = jax.random.permutation(permute_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        
        # Reshape into mini-batches for training
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )
        
        # Train on all mini-batches
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, train_key), transitions)

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            # gradient_steps is already updated in sgd_step
        )

        return (training_state, env_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, key):  
        @jax.jit
        def f(carry, unused_t):
            ts, es, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es), metrics = training_step(ts, es, train_key)
            return (ts, es, k), metrics

        (training_state, env_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, key), (), length=args.num_training_steps_per_epoch
        )
        
        return training_state, env_state, metrics

    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    training_walltime = 0
    print('Starting PQN training (NO REPLAY BUFFER)....')
    for ne in range(args.num_epochs):
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, metrics = training_epoch(
            training_state, env_state, epoch_key
        )
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)
        print(f"Epoch {ne} - step: {ne * args.num_training_steps_per_epoch * args.env_steps_per_actor_step / 1000000:.2f}M: =================================================")
        print(f" categorical_accuracy : {metrics['training/categorical_accuracy']*100:.2f} %")
        print(f" critic_loss : {metrics['training/critic_loss']:.4f}")
        print(f" logits_pos : {metrics['training/logits_pos']:.4f}")
        print(f" logits_neg : {metrics['training/logits_neg']:.4f}")
        print(f" temperature : {args.temperature:.4f} (fixed)")
        print(f" win_rate : {metrics['eval/win_rate']:.2f} %")
        print(f" =================================================\n")

        if args.checkpoint:
            params = training_state.critic_state.params
            path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
            save_params(path, params)
        
        if args.track:
            wandb.log(metrics, step=ne * args.num_training_steps_per_epoch * args.env_steps_per_actor_step)

            if args.wandb_mode == 'offline':
                trigger_sync()
    
    if args.checkpoint:
        params = training_state.critic_state.params
        path = f"{save_path}/final.pkl"
        save_params(path, params)

    print("Training complete!")

