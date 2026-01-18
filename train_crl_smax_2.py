
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

from brax import envs
from etils import epath
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue

import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = "crl_smax_2-sqrt-distance"
    seed: int = 1
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
    num_envs: int = 256
    num_eval_envs: int = 64
    critic_lr: float = 3e-4
    batch_size: int = 256
    rep_size: int = 64
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    
    # Temperature annealing parameters for exploration
    temperature_start: float = 1.0   # Start with high temperature (more exploration)
    temperature_end: float = 0.01    # End with low temperature (nearly greedy)
    temperature_decay_epochs: int = 100  # Decay over this many epochs

    max_replay_size: int = 5000
    min_replay_size: int = 1000
    
    unroll_length: int = 62

    # To be filled in runtime
    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0
    num_envs_agents: int = 0



# class SA_encoder(nn.Module):
#     """
#     State encoder that outputs representations for ALL actions at once.
#     Output shape: (batch, action_size * rep_size)
#     This allows implicit action selection via Q-value comparison.
#     """
#     action_size: int
#     rep_size: int
#     norm_type: str = "layer_norm"
    
#     @nn.compact
#     def __call__(self, s: jnp.ndarray):
#         lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
#         bias_init = nn.initializers.zeros
        
#         if self.norm_type == "layer_norm":
#             normalize = lambda x: nn.LayerNorm()(x)
#         else:
#             normalize = lambda x: x

#         x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(s)
#         x = normalize(x)
#         x = nn.swish(x)
#         x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
#         x = normalize(x)
#         x = nn.swish(x)
#         x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
#         x = normalize(x)
#         x = nn.swish(x)
#         x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
#         x = normalize(x)
#         x = nn.swish(x)
#         # Output: (batch, action_size * rep_size)
#         x = nn.Dense(self.action_size * self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
#         return x


class ActionHead(nn.Module):
    """Single action head - transforms shared features into action-specific representation."""
    rep_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        # Each action gets its own 2-layer MLP head
        x = nn.Dense(256, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)
        x = nn.swish(x)
        x = nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class SA_encoder(nn.Module):
    """
    State encoder with SEPARATE OUTPUT HEADS per action.
    
    Architecture:
    - Shared trunk processes state features
    - Each action has its OWN learnable head (not just a reshaped single output)
    
    This prevents representation collapse because each action has independent
    parameters in the final transformation, forcing diverse representations.
    
    Input: state (batch_size, obs_dim)
    Output: (batch_size, action_size, rep_size)
    """
    rep_size: int
    action_size: int
    norm_type: str = "layer_norm"
    
    @nn.compact
    def __call__(self, s: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        # Shared trunk - extract state features
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(s)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(512, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        # x is now (batch_size, 512) - shared features
        
        # Use nn.vmap to create action_size independent heads efficiently
        # - variable_axes={'params': 0} means each action gets its own params
        # - split_rngs={'params': True} ensures different init for each head
        # - This is more JAX-idiomatic than a Python for loop
        VmappedHead = nn.vmap(
            ActionHead,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None,  # Same input x for all heads
            out_axes=1,    # Stack outputs along axis 1
            axis_size=self.action_size,
        )
        
        # Single call: creates action_size heads, each with separate params
        # Output: (batch_size, action_size, rep_size)
        return VmappedHead(rep_size=self.rep_size, name="action_heads")(x)

class G_encoder(nn.Module):
    """Goal encoder - same as original."""
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
    """Contains training state for the learner - NO actor_state or alpha_state."""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    critic_state: TrainState
    temperature: jnp.ndarray  # Current temperature for exploration


class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray  # Now stores discrete action indices
    reward: jnp.ndarray
    discount: jnp.ndarray
    avail_actions: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))


if __name__ == "__main__":

    args = tyro.cli(Args)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps = int(np.ceil(args.min_replay_size / args.unroll_length))
    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)

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
    key, buffer_key, env_key, eval_env_key, sa_key, g_key = jax.random.split(key, 6)

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

    elif args.env_id == "smax_move":
        from envs.smax_move import SmaxEnv
        env = SmaxEnv()
        args.obs_dim = env.observation_size - 2
        args.goal_start_idx = env.observation_size - 4
        args.goal_end_idx = env.observation_size - 2
        args.num_agents = env.env.num_agents
        args.num_envs_agents = args.num_envs * args.num_agents
        args.action_size = env.action_size

    else:
        raise NotImplementedError(f"Environment {args.env_id} not supported in this script.")
    
    # Get available actions function for SMAX
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
    
    # Load checkpoint to resume training
    crtc_params = load_params(args.load_path) if args.load_path else None

    # Network setup - NO Actor, only Critic
    sa_encoder = SA_encoder(action_size=action_size, rep_size=args.rep_size)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]))
    
    g_encoder = G_encoder(rep_size=args.rep_size)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    
    if crtc_params is None:
        crtc_params = {"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}
    
    critic_state = TrainState.create(
        apply_fn=None,
        params=crtc_params,
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Training state - simplified, no actor or alpha
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        critic_state=critic_state,
        temperature=jnp.array(args.temperature_start),  # Start with high temperature
    )

    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((), dtype=jnp.int32)  # Discrete action index
    dummy_avail = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        avail_actions=dummy_avail,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "seed": 0.0,
            }        
        },
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer
    
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs_agents,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def compute_q_values(critic_params, obs, goal):
        """
        Compute Q-values for all actions given state and goal.
        Returns logits of shape (batch, action_size)
        """
        sa_encoder_params = critic_params["sa_encoder"]
        g_encoder_params = critic_params["g_encoder"]
        
        # Get state representations for all actions: (batch, action_size, rep_size)
        s_repr = sa_encoder.apply(sa_encoder_params, obs)
        s_repr = s_repr.reshape(-1, action_size, args.rep_size)
        
        # Get goal representation: (batch, rep_size) -> (batch, 1, rep_size)
        g_repr = g_encoder.apply(g_encoder_params, goal)
        g_repr = g_repr[:, None, :]
        
        # Compute negative L2 distance as Q-values: (batch, action_size)
        logits = -jnp.sqrt(jnp.sum((s_repr - g_repr) ** 2, axis=-1))
        # logits = -jnp.sum((s_repr - g_repr) ** 2, axis=-1)

        return logits

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        """Evaluation step - select actions greedily from Q-values."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        
        # Extract state and goal from observation
        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]
        
        # Compute Q-values for all actions
        logits = compute_q_values(training_state.critic_state.params, state, goal)
        
        # Mask unavailable actions
        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        logits = logits - ((1 - avail_actions) * 1e10)
        
        # Greedy action selection - discrete indices
        # actions = jnp.argmax(logits, axis=-1)
        actions =  jax.random.categorical(key, logits / 0.1, axis=-1)
        # Reshape for environment step: (num_envs, num_agents)
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

    def actor_step(critic_state, env, env_state, key, temperature, extra_fields):
        """Training step - sample actions from Q-values with temperature."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        
        # Extract state and goal from observation
        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]
        
        # Compute Q-values for all actions
        logits = compute_q_values(critic_state.params, state, goal)
        
        # Get available actions
        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        
        # Mask unavailable actions
        masked_logits = logits - ((1 - avail_actions) * 1e10)
        
        # Sample actions from softmax distribution with temperature
        # Higher temperature = more uniform (exploratory)
        # Lower temperature = more peaked (greedy)
        actions = jax.random.categorical(key, masked_logits / 0.1, axis=-1)
        
        # Reshape for environment step: (num_envs, num_agents)
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
    def get_experience(critic_state, env_state, buffer_state, key, temperature):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(critic_state, env, env_state, current_key, temperature, extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            # Use high temperature during prefill for more exploration
            env_state, buffer_state = get_experience(
                training_state.critic_state,
                env_state,
                buffer_state,
                key,
                temperature=args.temperature_start,  # High temperature for exploration
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            obs = transitions.observation[:, :args.obs_dim]
            action = transitions.action  # Discrete action indices
            
            # Get state representations for all actions: (batch, action_size, rep_size)
            sa_repr_all = sa_encoder.apply(sa_encoder_params, obs)
            sa_repr_all = sa_repr_all.reshape(-1, action_size, args.rep_size)
            
            # Select the representation for the taken action: (batch, rep_size)
            sa_repr = sa_repr_all[jnp.arange(sa_repr_all.shape[0]), action, :]
            
            # Get goal representation
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])
            
            # InfoNCE loss
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))  # shape = BxB
            # logits = -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)/0.1

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
        training_state = training_state.replace(critic_state=new_critic_state)

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
        training_state, key = carry
        key, critic_key = jax.random.split(key, 2)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        return (training_state, key), critic_metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # Update buffer with current temperature
        env_state, buffer_state = get_experience(
            training_state.critic_state,
            env_state,
            buffer_state,
            experience_key1,
            training_state.temperature,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        # Sample transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # Process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, args.obs_dim, args.goal_start_idx, args.goal_end_idx), transitions, batch_keys
        )
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
            transitions,
        )

        # Training steps
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):  
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (), length=args.num_training_steps_per_epoch
        )
        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)

    print('Prefilling replay buffer....')
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    # Setting up evaluator
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    training_walltime = 0
    print('Starting training....')
    for ne in range(args.num_epochs):
        t = time.time()

        # Compute current temperature with linear decay
        decay_progress = min(ne / args.temperature_decay_epochs, 1.0)
        current_temperature = args.temperature_start - (args.temperature_start - args.temperature_end) * decay_progress
        training_state = training_state.replace(temperature=jnp.array(current_temperature))

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key
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

        # Evaluation
        metrics = evaluator.run_evaluation(training_state, metrics)
        metrics["training/temperature"] = current_temperature
        
        # Print training metrics
        print(f"Epoch {ne} - step: {ne * args.num_training_steps_per_epoch * args.env_steps_per_actor_step / 1000000:.2f}M: Training Metrics: =================================================")
        print(f" temperature : {current_temperature:.4f}")
        print(f" categorical_accuracy : {metrics['training/categorical_accuracy']}")
        print(f" critic_loss : {metrics['training/critic_loss']:.4f}")
        print(f" logits_neg : {metrics['training/logits_neg']:.4f}")
        print(f" logits_pos : {metrics['training/logits_pos']:.4f}")
        print(f" win_rate : {metrics['eval/win_rate']*100:.2f} %")
        print(f" =================================================\n\n")
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

