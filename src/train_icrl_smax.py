import os
import jax
import flax
import hydra
import time
import copy
import optax
import pickle
import random
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import distrax

from brax import envs
from etils import epath
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from omegaconf import DictConfig, OmegaConf
from mava.utils.logger import MavaLogger, LogEvent
from mava.evaluator import CrlEvaluator
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue
from mava.wrappers.smax import SmaxEnv
from mava.utils.config import check_total_timesteps

# ICRL (Contrastive RL) implementation for SMAX
# This is the standalone version that uses Hydra config and Mava logger
# for direct comparison with the Mava implementation

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x
    
class G_encoder(nn.Module):
    norm_type = "layer_norm"
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(g)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(64, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x

class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"

    LOG_STD_MAX = 5
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)
        x = nn.Dense(1024, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        x = normalize(x)
        x = nn.swish(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState

class Transition(NamedTuple):
    """Container for a transition"""
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
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))


def run_experiment(config: DictConfig) -> float:
    """Main training loop."""
    
    # Make a deep copy to avoid modifying the original config
    config = copy.deepcopy(config)
    
    # Set logger system name
    config.logger.system_name = "icrl_smax"
    
    # Compute derived values
    config.system.env_steps_per_actor_step = config.system.num_envs * config.system.unroll_length
    config.system.num_prefill_env_steps = config.system.min_replay_size * config.system.num_envs
    config.system.num_prefill_actor_steps = int(np.ceil(config.system.min_replay_size / config.system.unroll_length))
    config.system.num_training_steps_per_epoch = (
        (config.system.total_env_steps - config.system.num_prefill_env_steps) // 
        (config.system.num_epochs * config.system.env_steps_per_actor_step)
    )
    
    run_name = f"{config.env.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    # Environment setup    
    if config.env.env_id == "smax":
        env = SmaxEnv(map_name=config.env.smax_map_name)
        
        config.env.obs_dim = env.observation_size - 1
        config.env.goal_start_idx = env.observation_size - 2
        config.env.goal_end_idx = env.observation_size - 1
        config.env.num_agents = env.env.num_agents
        config.system.num_envs_agents = config.system.num_envs * config.env.num_agents
    else:
        raise NotImplementedError
    
    # Get available actions function for SMAX
    if 'smax' in config.env.env_id:
        get_avail_act = jax.vmap(env.get_avail_actions)

    env = envs.training.wrap(
        env,
        episode_length=config.env.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    env_keys = jax.random.split(env_key, config.system.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    # Load checkpoint to resume training (if specified)
    if config.checkpoint.load_path:
        alph_params, act_params, crtc_params = load_params(config.checkpoint.load_path)
    else:
        alph_params, act_params, crtc_params = None, None, None

    # Network setup
    # Actor
    actor = Actor(action_size=action_size)
    if not act_params:
        act_params = actor.init(actor_key, np.ones([1, obs_size]))
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=act_params,
        tx=optax.adam(learning_rate=config.system.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder()
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, config.env.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder()
    g_encoder_params = g_encoder.init(g_key, np.ones([1, config.env.goal_end_idx - config.env.goal_start_idx]))
    c = jnp.asarray(0.0, dtype=jnp.float32)
    if not crtc_params:
        crtc_params = {"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
        tx=optax.adam(learning_rate=config.system.critic_lr),
    )

    # Entropy coefficient
    target_entropy = -0.5 * action_size
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    if not alph_params:
        alph_params = {"log_alpha": log_alpha}
    alpha_state = TrainState.create(
        apply_fn=None,
        params=alph_params,
        tx=optax.adam(learning_rate=config.system.alpha_lr),
    )
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    # Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))

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

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer
    
    # Change replay buffer to make the 2nd dimension num_envs_agents
    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.system.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=config.system.batch_size,
            num_envs=config.system.num_envs_agents,
            episode_length=config.env.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    # Actor step functions (modified for multiple agents)
    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        means, _ = actor.apply(training_state.actor_state.params, obs)
        action = None
        avail_actions = None
        
        if not config.env.discrete_actions:
            if 'smax' in config.env.env_id:
                avail_actions = get_avail_act(env_state)
                avail_actions = jnp.reshape(avail_actions, (-1,)+avail_actions.shape[2:])
                means = means - ((1-avail_actions)*1e10)
                
            actions = nn.tanh(means)
        else:
            actions = jnp.argmax(means, axis=-1)
        
        actions_ = jnp.reshape(actions, (-1, config.env.num_agents,) + actions.shape[1:])
        nstate = env.step(env_state, actions_)
        
        state_extras = {x: jnp.repeat(nstate.info[x], config.env.num_agents) for x in extra_fields}
        return nstate, Transition(
            observation=obs,
            action=actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, config.env.num_agents),
            discount=jnp.repeat(1-nstate.done, config.env.num_agents),
            extras={"state_extras": state_extras},
        )

    def actor_step(actor_state, env, env_state, key, extra_fields):
        # Perform inference to get actions
        keys = jax.random.split(key, config.env.num_agents)
        obs = jnp.reshape(env_state.obs, (-1,) + env_state.obs.shape[2:])
        means, log_stds = actor.apply(actor_state.params, obs)
        actions = None
        transition_actions = None
        avail_actions = None
        
        if not config.env.discrete_actions:
            stds = jnp.exp(log_stds)
            noise = jax.vmap(lambda rng: jax.random.gumbel(
                        rng, shape=(config.system.num_envs,) + means.shape[1:], 
                        dtype=means.dtype), in_axes=0, out_axes=1)(keys)
            noise_ = jnp.reshape(noise, (-1,) + noise.shape[2:])
            
            if 'smax' in config.env.env_id:
                avail_actions = get_avail_act(env_state)
                avail_actions = jnp.reshape(avail_actions, (-1,)+avail_actions.shape[2:])
                means = means - ((1-avail_actions)*1e10)
            
            actions = nn.tanh(means + stds * noise_)
            transition_actions = actions
        else:
            avail_actions = get_avail_act(env_state)
            avail_actions = jnp.reshape(avail_actions, (-1,)+avail_actions.shape[2:])
            action_logits = means - ((1-avail_actions)*1e10)
            
            pi = distrax.Categorical(logits = action_logits)
            actions = pi.sample(seed = keys[0])
            transition_actions = jax.nn.one_hot(actions, means.shape[1])
            
        # Step environment
        actions_ = jnp.reshape(actions, (-1, config.env.num_agents,) + actions.shape[1:])
        nstate = env.step(env_state, actions_)
        
        # Generate an array of transitions, with shape (num_envs*num_agents,...)
        state_extras = {x: jnp.repeat(nstate.info[x], config.env.num_agents) for x in extra_fields}

        return nstate, Transition(
            observation=obs,
            action=transition_actions,
            avail_actions=avail_actions,
            reward=jnp.repeat(nstate.reward, config.env.num_agents),
            discount=jnp.repeat(1-nstate.done, config.env.num_agents),
            extras={"state_extras": state_extras},
        )

    @jax.jit
    def get_experience(actor_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(actor_state, env, env_state, current_key, extra_fields=("truncation", "seed"))
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=config.system.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                env_state,
                buffer_state,
                key,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + config.system.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=config.system.num_prefill_actor_steps)[0]

    # Update functions (modified to accommodate discrete action space)
    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, :config.env.obs_dim]
            future_state = transitions.extras["future_state"]
            goal = future_state[:, config.env.goal_start_idx : config.env.goal_end_idx]
            observation = jnp.concatenate([state, goal], axis=1)
            avail_actions = transitions.avail_actions

            means, log_stds = actor.apply(actor_params, observation)
            action = None
            log_prob = None
            
            if not config.env.discrete_actions:
                stds = jnp.exp(log_stds)
                means = means - ((1-avail_actions)*1e10)
                x_ts = means + stds * jax.random.gumbel(key, shape=means.shape, dtype=means.dtype)
                action = nn.tanh(x_ts)
                x_std = (x_ts - means)/stds
                log_prob = -(x_std + jnp.exp(-x_std)) - log_stds
                
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = jnp.where(avail_actions == 0, 0, log_prob)
                log_prob = log_prob.sum(-1)
            else:
                pi = distrax.Categorical(logits = means)
                action = pi.sample(seed = key)
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
        
        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params, training_state.critic_state.params, 
            training_state.alpha_state.params['log_alpha'], transitions, key
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
        new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

        training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alpha_loss": alphaloss,   
            "log_alpha": training_state.alpha_state.params["log_alpha"],
        }

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, transitions, key):
            sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
            
            obs = transitions.observation[:, :config.env.obs_dim]
            action = transitions.action
            
            sa_repr = sa_encoder.apply(sa_encoder_params, obs, action)
            g_repr = g_encoder.apply(g_encoder_params, transitions.observation[:, config.env.obs_dim:])
            
            # InfoNCE
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            # Logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            critic_loss += config.system.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

            return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)
            
        (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(training_state.critic_state.params, transitions, key)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state = new_critic_state)

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
        key, critic_key, actor_key, = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        return (training_state, key,), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # Update buffer
        env_state, buffer_state = get_experience(
            training_state.actor_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + config.system.env_steps_per_actor_step,
        )

        # Sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # Process transitions for training (HER-style goal relabeling)
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (config.system.gamma, config.env.obs_dim, config.env.goal_start_idx, config.env.goal_end_idx), 
            transitions, batch_keys
        )
        
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
            transitions,
        )

        # Take actor-step worth of training-step
        (training_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        return (training_state, env_state, buffer_state,), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        key,
    ):  
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (), 
            length=config.system.num_training_steps_per_epoch
        )
        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    # Prefill replay buffer
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    # Setting up evaluator
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        env,
        num_eval_envs=config.system.num_eval_envs,
        episode_length=config.env.episode_length,
        key=eval_env_key,
    )
    
    # Initialize Mava logger
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    
    # Checkpointing setup
    save_checkpoint = config.checkpoint.enabled
    checkpoint_dir = None
    if save_checkpoint:
        from pathlib import Path
        checkpoint_dir = Path(config.logger.base_exp_path).expanduser() / config.logger.system_name / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")

    training_walltime = 0
    max_win_rate = -jnp.inf
    print('Starting training....')
    
    for ne in range(config.system.num_epochs):
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key
        )
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (config.system.env_steps_per_actor_step * config.system.num_training_steps_per_epoch) / epoch_training_time
        env_steps = int(training_state.env_steps.item())
        
        train_metrics = {
            "steps_per_second": sps,
            "walltime": training_walltime,
            "env_steps": env_steps,
            **{name: float(value) for name, value in metrics.items()},
        }

        # Run evaluation and get metrics
        eval_metrics_dict = evaluator.run_evaluation(training_state, {})
        eval_metrics = {
            name: float(value) for name, value in eval_metrics_dict.items()
        }
        
        win_rate = eval_metrics.get("win_rate", 0.0)
        episode_return = eval_metrics.get("episode_return", 0.0)
        
        # Log with Mava logger
        logger.log({"timestep": env_steps}, env_steps, ne, LogEvent.MISC)
        logger.log(train_metrics, env_steps, ne, LogEvent.TRAIN)
        logger.log(eval_metrics, env_steps, ne, LogEvent.EVAL)
        
        
        # Save checkpoints
        if save_checkpoint and checkpoint_dir:
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                params = (
                    training_state.alpha_state.params, 
                    training_state.actor_state.params, 
                    training_state.critic_state.params
                )
                best_path = checkpoint_dir / "best_model.pkl"
                save_params(str(best_path), params)
                print(f"  Saved best model (win_rate={win_rate:.2%})")
    
    # Save final checkpoint
    if save_checkpoint and checkpoint_dir:
        params = (
            training_state.alpha_state.params, 
            training_state.actor_state.params, 
            training_state.critic_state.params
        )
        final_path = checkpoint_dir / "final_model.pkl"
        save_params(str(final_path), params)
        print(f"Saved final model to {final_path}")
    
    # Stop logger
    logger.stop()
    
    # Return final win rate for hyperparameter tuning
    return float(eval_metrics.get("win_rate", 0.0))


@hydra.main(
    config_path=".",
    config_name="config_icrl_smax",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)
    
    try:
        eval_performance = run_experiment(cfg)
        return eval_performance
    except Exception as e:
        print(f"Error executing experiment")
        print(f"Exception: {type(e).__name__}: {e!s}")
        import traceback
        traceback.print_exc()
        return float("-inf")


if __name__ == "__main__":
    hydra_entry_point()
    
        