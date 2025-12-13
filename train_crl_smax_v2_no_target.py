"""
CRL-DQN with Explicit Action Encoding - NO TARGET NETWORK

This is an ablation to test if the target network (soft update with tau) 
has any effect on performance. We use only one network for both 
action selection and training.

Key difference from v2: No target_critic_params, no soft_update.
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
    exp_name: str = "crl_v2_no_target"
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
    
    # NO target_tau - we don't use target network!

    # Learned temperature parameters (like SAC's alpha)
    temperature_lr: float = 3e-4
    initial_temperature: float = 0.3
    min_temperature: float = 0.01
    target_entropy_ratio: float = 0.1

    max_replay_size: int = 5000
    min_replay_size: int = 1000
    
    unroll_length: int = 62

    # To be filled in runtime
    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0
    num_envs_agents: int = 0


class SA_encoder(nn.Module):
    """
    State-Action encoder that takes (state, action_onehot) as input.
    """
    rep_size: int
    norm_type: str = "layer_norm"
    
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
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
    """Goal encoder."""
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
    """Contains training state - NO target_critic_params!"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    critic_state: TrainState
    # NO target_critic_params here!
    temperature_state: TrainState


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


# NO soft_update function needed!


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
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Temperature learning
    max_entropy = jnp.log(action_size)
    target_entropy = args.target_entropy_ratio * max_entropy
    print(f"Target entropy: {target_entropy:.4f} (max: {max_entropy:.4f})")
    print("NOTE: Running WITHOUT target network (no soft update)")
    
    log_temperature = jnp.asarray(jnp.log(args.initial_temperature), dtype=jnp.float32)
    temperature_state = TrainState.create(
        apply_fn=None,
        params={"log_temperature": log_temperature},
        tx=optax.adam(learning_rate=args.temperature_lr),
    )

    # Training state - NO target_critic_params!
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        critic_state=critic_state,
        # NO target_critic_params!
        temperature_state=temperature_state,
    )

    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((), dtype=jnp.int32)
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

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        """Evaluation step - greedy action selection using ONLINE params."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        
        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]
        
        # Use online params directly (no target network)
        logits = compute_q_values(training_state.critic_state.params, state, goal)
        
        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        logits = logits - ((1 - avail_actions) * 1e10)
        
        actions = jnp.argmax(logits, axis=-1)
        
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

    def actor_step(critic_params, env, env_state, key, log_temperature, extra_fields):
        """Training step - sample actions using ONLINE params."""
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])

        state = obs[:, :args.obs_dim]
        goal = obs[:, args.obs_dim:]

        # Use online params directly (no target network)
        logits = compute_q_values(critic_params, state, goal)

        avail_actions = get_avail_act(env_state)
        avail_actions = jnp.reshape(avail_actions, (-1,) + avail_actions.shape[2:])
        masked_logits = logits - ((1 - avail_actions) * 1e10)

        temperature = jnp.exp(log_temperature)
        actions = jax.random.categorical(key, masked_logits / temperature, axis=-1)

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
    def get_experience(critic_params, env_state, buffer_state, key, log_temperature):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)

            env_state, transition = actor_step(
                critic_params,  # Use online params
                env,
                env_state,
                current_key,
                log_temperature,
                extra_fields=("truncation", "seed"),
            )
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
            env_state, buffer_state = get_experience(
                training_state.critic_state.params,  # Use online params
                env_state,
                buffer_state,
                key,
                training_state.temperature_state.params["log_temperature"],
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
            action = transitions.action
            
            action_onehot = jax.nn.one_hot(action, action_size)
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action_onehot)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, args.obs_dim:])
            
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

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

        # NO soft_update! Just update the critic directly
        training_state = training_state.replace(
            critic_state=new_critic_state,
            # NO target_critic_params update!
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
    def update_temperature(transitions, training_state):
        """Update temperature to maintain target entropy."""
        def temperature_loss(temp_params, transitions):
            log_temperature = temp_params["log_temperature"]
            log_temperature = jnp.maximum(log_temperature, jnp.log(args.min_temperature))
            temperature = jnp.exp(log_temperature)
            
            obs = transitions.observation[:, :args.obs_dim]
            goal = transitions.observation[:, args.obs_dim:]
            avail_actions = transitions.avail_actions
            
            q_values = compute_q_values(training_state.critic_state.params, obs, goal)
            
            masked_q = q_values - ((1 - avail_actions) * 1e10)
            
            policy = jax.nn.softmax(masked_q / temperature, axis=-1)
            
            log_policy = jnp.log(policy + 1e-8)
            entropy = -jnp.sum(policy * log_policy * avail_actions, axis=-1)
            
            temp_loss = log_temperature * jnp.mean(entropy - target_entropy)
            
            return temp_loss, entropy
        
        (temp_loss, entropy), temp_grad = jax.value_and_grad(temperature_loss, has_aux=True)(
            training_state.temperature_state.params, transitions
        )
        new_temperature_state = training_state.temperature_state.apply_gradients(grads=temp_grad)
        
        clamped_log_temp = jnp.maximum(
            new_temperature_state.params["log_temperature"], 
            jnp.log(args.min_temperature)
        )
        new_temperature_state = new_temperature_state.replace(
            params={"log_temperature": clamped_log_temp}
        )
        
        training_state = training_state.replace(temperature_state=new_temperature_state)
        
        metrics = {
            "policy_entropy": jnp.mean(entropy),
            "temperature": jnp.exp(training_state.temperature_state.params["log_temperature"]),
            "temperature_loss": temp_loss,
        }
        
        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key = jax.random.split(key, 2)

        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state, temp_metrics = update_temperature(transitions, training_state)

        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(critic_metrics)
        metrics.update(temp_metrics)
        
        return (training_state, key), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)
        
        env_state, buffer_state = get_experience(
            training_state.critic_state.params,  # Use online params
            env_state,
            buffer_state,
            experience_key1,
            training_state.temperature_state.params["log_temperature"],
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        buffer_state, transitions = replay_buffer.sample(buffer_state)

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

    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    training_walltime = 0
    print('Starting training (v2 NO TARGET NETWORK)....')
    for ne in range(args.num_epochs):
        t = time.time()

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

        metrics = evaluator.run_evaluation(training_state, metrics)
        print(f"Epoch {ne} - step: {ne * args.num_training_steps_per_epoch * args.env_steps_per_actor_step / 1000000:.2f}M: =================================================")
        print(f" categorical_accuracy : {metrics['training/categorical_accuracy']*100:.2f} %")
        print(f" critic_loss : {metrics['training/critic_loss']:.4f}")
        print(f" logits_pos : {metrics['training/logits_pos']:.4f}")
        print(f" logits_neg : {metrics['training/logits_neg']:.4f}")
        print(f" temperature : {metrics['training/temperature']:.4f} (target: {target_entropy:.2f})")
        print(f" policy_entropy : {metrics['training/policy_entropy']:.4f}")
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

