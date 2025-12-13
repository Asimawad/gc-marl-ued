"""
This implementation of MEGA (GoalKDE) adapts train_crl_latent_actor_kde.py (minLSE goal selection), and as of this comment, differs only by the following components:
- Adds KDE fitting (with default bandwidth 0.1 and Gaussian kernel, as recommended)
- Uses the KDE to select sampled goals with the lowest probability under the KDE
"""

import sys
import os

# Get the absolute path to the parent directory of navix/
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add parent directory to sys.path
sys.path.append(parent_dir)

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
import navix as nx
import seaborn as sns
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt

from etils import epath
from dataclasses import dataclass
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState

from buffer import TrajectoryUniformSamplingQueue

from jax.scipy import stats

@dataclass
class Args:
    #environment specific arguments
    env_id: str = "gcrl_empty"
    # to be filled in runtime
    episode_length: int = 0
    goal_start_channel: int = 0
    goal_end_channel: int = 1

    # exp_name: str = os.path.basename(__file__)[len("train_"): -len(".py")]
    alg_name: str = os.path.basename(__file__)[len("train_"): -len(".py")]
    exp_no: str = "test"
    # seed: int = 1
    n_seeds: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "sampling_goals"
    wandb_entity: str = 'ahmed-turkman-aims-south-africa'
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    capture_video: bool = False
    checkpoint: bool = False


    # Algorithm specific arguments
    total_env_steps: int = 50000000
    num_epochs: int = 25
    num_envs: int = 256
    num_eval_envs: int = 128
    critic_lr: float = 3e-4
    batch_size: int = 256
    rep_size: int = 64
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 8000
    min_replay_size: int = 1000
    
    unroll_length: int  = 100

    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    critic_state: TrainState
    xy_coverage: jnp.ndarray
    kde: jax._src.scipy.stats.kde.gaussian_kde
       
class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
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
    args.num_prefill_actor_steps = np.ceil(args.min_replay_size / args.unroll_length)
    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)

    wandb_group: str = "Navix_" + args.env_id + '_' + args.alg_name + '_' + args.exp_no

    
    for seed in range(args.n_seeds):

        print('\n' + '-' * 80)
        print('starting a new run...')
        print('seed =', seed)
        print('-' * 80, end='\n\n')

        # run_name = f"{args.env_id}__{args.alg_name}__{seed}__{int(time.time())}"
        run_name = f"{args.env_id}_{args.alg_name}_{args.exp_no}_{seed}_{int(time.time())}"

        if args.track:

            if wandb_group ==  '.':
                wandb_group = 'normal'
                
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                group=wandb_group,
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

        random.seed(seed)
        np.random.seed(seed)
        key = jax.random.PRNGKey(seed)
        key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

        # Environment setup
        if args.env_id == "gcrl_four_rooms":
            from envs.nx_four_rooms import gcrl_FourRooms, four_rooms_goal_coords as env_goal_coords
            env = nx.make('gcrl_four_rooms')
            grid_shape = (16,16)

        elif args.env_id == "gcrl_empty":
            from envs.nx_empty import gcrl_empty, empty_goal_coords as env_goal_coords
            env = nx.make('gcrl_empty')
            grid_shape = (16,16)
        
        elif args.env_id == "gcrl_door_key":
            from envs.nx_door_key import gcrl_DoorKey, door_key_goal_coords as env_goal_coords
            env = nx.make('gcrl_door_key')
            grid_shape = (16,16)
            
        elif args.env_id == 'Navix-KeyCorridorS3R1-v0':
            from envs.nx_key_corridor import gcrl_KeyCorridor, key_corridor_S3R1_coords as env_goal_coords
            env = nx.make('Navix-KeyCorridorS3R1-v0')
            grid_shape = (3,7)

        elif args.env_id == 'Navix-KeyCorridorS3R2-v0':
            from envs.nx_key_corridor import gcrl_KeyCorridor, key_corridor_S3R2_coords as env_goal_coords
            env = nx.make('Navix-KeyCorridorS3R2-v0')
            grid_shape = (5,7)

        elif args.env_id == 'Navix-KeyCorridorS3R3-v0':
            from envs.nx_key_corridor import gcrl_KeyCorridor, key_corridor_S3R3_coords as env_goal_coords
            env = nx.make('Navix-KeyCorridorS3R3-v0')
            grid_shape = (7,7)

        elif args.env_id == 'Navix-KeyCorridorS4R3-v0':
            from envs.nx_key_corridor import gcrl_KeyCorridor, key_corridor_S4R3_coords as env_goal_coords
            env = nx.make('Navix-KeyCorridorS4R3-v0', max_steps=300)
            grid_shape = (10,10)

        else:
            raise NotImplementedError
        
        @jax.jit
        def env_step_jit(timestep, action):
            return env.step(timestep, action)
        
        @jax.jit
        def env_reset_jit(env_keys):
            return env.reset(env_keys)


        env_keys = jax.random.split(env_key, args.num_envs)
        env_reset_jit_vmap = jax.vmap(env_reset_jit)
        env_step_jit_vmap = jax.vmap(env_step_jit)
        env_state = env_reset_jit_vmap(env_keys)

        # concrete values
        num_envs = int(args.num_envs)
        action_size = int(env.action_space.n)
        rep_size = int(args.rep_size)
        max_steps = int(env.max_steps)

        # Critic
        from models import sa_ConvEncoder, small_G_encoder as G_Encoder

        sa_encoder = sa_ConvEncoder(output_size=env.action_space.n * args.rep_size) # Why is it multiplied by the action size?
        sa_encoder_params = sa_encoder.init(sa_key, jnp.ones([1, *env.observation_space.shape])) # Why is there a 1 in the beginning?
        g_encoder = G_Encoder(rep_size=args.rep_size)
        g_encoder_params = g_encoder.init(g_key, jnp.ones(2))

        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=args.critic_lr),
        )

        sa_encoder.apply = jax.jit(sa_encoder.apply)
        g_encoder.apply = jax.jit(g_encoder.apply)

        dummy_goal_batch = jnp.zeros([10000, 2])
        dummy_kde = stats.gaussian_kde(dummy_goal_batch.T, bw_method=0.1) # We'll fit this properly before the first time it's used

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            critic_state=critic_state,
            xy_coverage=jnp.zeros(shape=grid_shape),
            kde=dummy_kde,
        )

        #Replay Buffer
        dummy_goal = jnp.zeros((2,), dtype=jnp.int32)
        dummy_obs = jnp.zeros((*env.observation_space.shape,), dtype=env.observation_space.dtype)
        dummy_action = jnp.zeros((1,), dtype=env.action_space.dtype)

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            extras={
                "state_extras": {
                    "seed": 0,
                    "goal": dummy_goal,
                }        
            },
        )

        def jit_wrap(buffer):
            buffer.insert = jax.jit(buffer.insert)
            buffer.sample = jax.jit(buffer.sample)
            return buffer
        
        replay_buffer = jit_wrap(
                TrajectoryUniformSamplingQueue(
                    max_replay_size=args.max_replay_size,
                    dummy_data_sample=dummy_transition,
                    sample_batch_size=args.batch_size,
                    num_envs=args.num_envs,
                    sequence_length=args.num_envs+1,
                )
            )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        def deterministic_actor_step(training_state, env_state, g_repr, key, extra_fields):
            """Function to collect data during evaluation. Used in evaluator.py"""
            state = env_state.observation
            s_repr = sa_encoder.apply(training_state.critic_state.params["sa_encoder"], state).reshape(-1, action_size, args.rep_size)

            logits = -jnp.sqrt( jnp.sum( ( s_repr - g_repr ) ** 2, axis=-1) )
            actions =  jax.random.categorical(key, logits / 0.1, axis=-1)
            # actions = jnp.argmax(logits, axis=-1)

            nstate = env_step_jit_vmap(env_state, actions)

            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.observation,
                action=actions,
                extras={
                    'state_extras': state_extras
                })
        
        def run_evaluation(training_state, training_metrics, key, aggregate_episodes = True):
            key, unroll_key = jax.random.split(key)
            t = time.time()

            def generate_unroll(training_state, env_state, g_repr, actor_key, unroll_length, extra_fields=()):
                """Collect trajectories of given unroll_length."""
                @jax.jit
                def f(carry, unused_t):
                    state, key = carry
                    current_key, key = jax.random.split(key)
                    nstate, transition = deterministic_actor_step(training_state, state, g_repr, current_key, extra_fields=extra_fields)
                    return (nstate, key), transition

                (final_state, _), data = jax.lax.scan(f, (env_state, actor_key), (), length=unroll_length)
                return final_state, data
            
            @jax.jit
            def generate_eval_unroll(training_state, key):
                unroll_key, actor_key, key = jax.random.split(key, 3)
                reset_keys = jax.random.split(unroll_key, args.num_eval_envs)
                eval_first_state = env_reset_jit_vmap(reset_keys)

                g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], env_goal_coords)

                eval_final_state, _ =  generate_unroll(
                                        training_state,
                                        eval_first_state,
                                        g_repr, 
                                        actor_key,
                                        unroll_length=max_steps)
                return eval_first_state, eval_final_state
            
            eval_first_state, eval_state = generate_eval_unroll(training_state, unroll_key)
            steps_per_unroll = max_steps * args.num_eval_envs

            eval_metrics = eval_state.info       
            epoch_eval_time = time.time() - t
            metrics = {}
            aggregating_fns = [(np.mean, ""),]

            for (fn, suffix) in aggregating_fns:
                metrics.update(
                    {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics[name]) if aggregate_episodes else eval_metrics[name]
                    )
                    for name in ['return'] if name in eval_metrics.keys()
                    }
                )
            
            # We check in how many env there was at least one step where there was success
            if "return" in eval_metrics:
                metrics["eval/episode_success_any"] = np.mean(
                    eval_metrics["return"] > 0.0
                )

            # We check the total distance travelled on average
            metrics["eval/distance_from_start"] = np.mean(
                jnp.sum(jnp.abs(eval_first_state.state.entities['player'].position - eval_state.state.entities['player'].position), axis=-1)
                )
            
            # We also check the total distance between the final state and the goal
            metrics["eval/distance_from_goal"] = np.mean(
                jnp.sum(jnp.abs(eval_state.state.entities['player'].position - env_goal_coords), axis=-1)
                )
            
            metrics["eval/avg_episode_length"] = np.mean(eval_state.t)
            metrics["eval/epoch_eval_time"] = epoch_eval_time
            metrics["eval/sps"] = steps_per_unroll / epoch_eval_time

            training_metrics.update(metrics)

            return training_metrics, key
        
        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, key = carry
                key, current_key = jax.random.split(key)

                state = env_state.observation
                s_repr = sa_encoder.apply(training_state.critic_state.params["sa_encoder"], state).reshape(num_envs, action_size, rep_size)      
                g_repr = g_encoder.apply(training_state.critic_state.params["g_encoder"], env_goal_coords)

                logits = -jnp.sqrt( jnp.sum( ( s_repr - g_repr ) ** 2, axis=-1) )
                actions =  jax.random.categorical(key, logits, axis=-1)

                nstate = env_step_jit_vmap(env_state, actions)

                training_state = training_state.replace(
                    xy_coverage = training_state.xy_coverage.at[(nstate.state.entities['player'].position[:, 0, 0], nstate.state.entities['player'].position[:, 0, 1] )].add(1)
                )

                state_extras = {x: nstate.info[x] for x in ("seed",)}
                state_extras['goal'] = nstate.state.entities['player'].position

                transition = Transition(
                    observation=env_state.observation,
                    action=actions,
                    extras={"state_extras": state_extras},
                )
    
                return (training_state, nstate, key), (transition)

            (training_state, env_state, _), data =  jax.lax.scan(f, (training_state, env_state, key), (), length=args.unroll_length * args.num_prefill_actor_steps)
            
            buffer_state = replay_buffer.insert(buffer_state, data)
            training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step * args.num_prefill_actor_steps)
            
            return training_state, env_state, buffer_state

        def actor_step(training_state, env_state, explore_goal_rep, explore_goals, key, extra_fields):
        
            alpha = jnp.where(
                jnp.all(explore_goals == env_state.state.entities['player'].position[:,0,:], axis=-1),
                10,
                0.1,
            )
            alpha = jnp.tile(alpha[:, None], (1, action_size))

            state = env_state.observation
            s_repr = sa_encoder.apply(training_state.critic_state.params["sa_encoder"], state).reshape(-1, action_size, args.rep_size)
            g_reps = jnp.repeat(explore_goal_rep[:, None, :], action_size, axis=1)
            logits = -jnp.sqrt( jnp.sum( ( s_repr - g_reps ) ** 2, axis=-1) )

            actions =  jax.random.categorical(key, logits / alpha, axis=-1)

            nstate = env_step_jit_vmap(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras['goal'] = nstate.state.entities['player'].position

            training_state = training_state.replace(
                    xy_coverage = training_state.xy_coverage.at[(nstate.state.entities['player'].position[:, 0, 0], nstate.state.entities['player'].position[:, 0, 1] )].add(1)
                )
            
            return training_state, nstate, Transition(
                                                observation=env_state.observation,
                                                action=actions,
                                                extras={"state_extras": state_extras},
                                            )
        
        @jax.jit
        def get_experience(training_state, transitions, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                training_state, env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                training_state, env_state, transition = actor_step(training_state, env_state, exploration_goal_representations, exploration_goals, current_key, extra_fields=("seed",))
                return (training_state, env_state, next_key), transition

            # @jax.jit
            # def get_worst_best_goal(rs, ra, rg):
            #     sa_encoder_params, g_encoder_params = jax.lax.stop_gradient(training_state.critic_state.params["sa_encoder"]), jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])
            #     random_sa_repr = sa_encoder.apply(sa_encoder_params, rs).reshape(num_envs, action_size, rep_size)
            #     random_sa_repr = random_sa_repr[jnp.arange(random_sa_repr.shape[0]), ra.squeeze(-1), :]
            #     random_g_repr = g_encoder.apply(g_encoder_params, rg)            
            #     logits = -jnp.sqrt(jnp.sum((random_sa_repr[:, None, :] - random_g_repr[None, :, :]) ** 2, axis=-1))
            #     return random_g_repr[ jnp.argmin( jax.nn.logsumexp(logits + 1e-6, axis=0) ) ], rg[ jnp.argmin( jax.nn.logsumexp(logits + 1e-6, axis=0) ) ]
            @jax.vmap
            def get_lowest_probability_goal(buffer_goals):
                return buffer_goals[jnp.argmin(training_state.kde(buffer_goals.T))]


            # random_g_key, key = jax.random.split(key, 2)
            
            # # sampling random state, actions
            # random_states = transitions.observation.reshape(args.num_envs, args.num_envs, *env.observation_space.shape)
            # random_actions = transitions.action.reshape(args.num_envs, args.num_envs, -1)
            
            # permutation = jax.random.permutation(random_g_key, len(transitions.extras['goal']))
            # random_goals = transitions.extras['goal'][permutation].reshape(args.num_envs, args.num_envs, 2)

            # explore_goal_reps, explore_goals = jax.vmap(get_worst_best_goal, in_axes=0)(random_states, random_actions, random_goals)


            # Permute and reshape buffer goals (from flat to matrix)
            key, subkey = jax.random.split(key)
            buffer_goals = jax.random.permutation(subkey, transitions.extras['goal'])
            buffer_goals = buffer_goals.reshape(args.num_envs, args.num_envs, -1) # Permute and expand back to (#envs, #envs, goal_dim.
                                                                                # first dim is for vectorized envs, second dim is just # of samples
                                                                                # which we happen to make equal to num_envs...?
                    
            # Get goal representation prescribed by exploration rule (lowest probability goal under KDE, then convert to goal representation)
            exploration_goals = get_lowest_probability_goal(buffer_goals)
            g_encoder_params = jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])
            exploration_goal_representations = g_encoder.apply(g_encoder_params, exploration_goals)


            (training_state, env_state, _), data = jax.lax.scan(f, (training_state, env_state, key), (), length=args.unroll_length)

            buffer_state = replay_buffer.insert(buffer_state, data)
            return training_state, env_state, buffer_state

        # @jax.jit
        # def get_worst_best_goal(carry, unused_t):
        #     init_eid, random_states, random_actions, random_goals, key = carry 
        #     rs = jax.lax.dynamic_slice_in_dim( random_states, init_eid * (args.num_envs),  (args.num_envs) )
        #     ra = jax.lax.dynamic_slice_in_dim( random_actions, init_eid * (args.num_envs),  (args.num_envs) )
        #     rg = jax.lax.dynamic_slice_in_dim( random_goals, init_eid * (args.num_envs),  (args.num_envs) )

        #     sa_encoder_params, g_encoder_params = jax.lax.stop_gradient(training_state.critic_state.params["sa_encoder"]), jax.lax.stop_gradient(training_state.critic_state.params["g_encoder"])
        #     random_sa_repr = sa_encoder.apply(sa_encoder_params, rs).reshape(num_envs, action_size, rep_size)
        #     random_sa_repr = random_sa_repr[jnp.arange(random_sa_repr.shape[0]), ra, :]

        #     random_g_repr = g_encoder.apply(g_encoder_params, rg)            
        #     logits = -jnp.sqrt(jnp.sum((random_sa_repr[:, None, :] - random_g_repr[None, :, :]) ** 2, axis=-1))
        #     return (init_eid + 1, random_states, random_actions, random_goals, key), random_g_repr[ jnp.argmin( jax.nn.logsumexp(logits + 1e-6, axis=0) ) ]
        
        # @jax.jit
        # def get_experience(training_state, transitions, env_state, buffer_state, key):
        #     @jax.jit
        #     def f(carry, unused_t):
        #         training_state, env_state, current_key = carry
        #         current_key, next_key = jax.random.split(current_key)
        #         training_state, env_state, transition = actor_step(training_state, env_state, explore_goal_reps, current_key, extra_fields=("seed",))
        #         return (training_state, env_state, next_key), transition

        #     random_g_key, key = jax.random.split(key, 2)
            
        #     permutation = jax.random.permutation(random_g_key, len(transitions.extras['goal']))
        #     random_goals = transitions.extras['goal'][permutation] #.reshape(args.num_envs, args.num_envs, 2)

        #     init_eid = 0
        #     (_, _, _, _, key), explore_goal_reps = jax.lax.scan(get_worst_best_goal, (init_eid, transitions.observation, transitions.action, random_goals, key), (), length=args.num_envs)

        #     (training_state, env_state, _), data = jax.lax.scan(f, (training_state, env_state, key), (), length=args.unroll_length)

        #     buffer_state = replay_buffer.insert(buffer_state, data)
        #     return training_state, env_state, buffer_state

        @jax.jit
        def update_critic(transitions, training_state, key):
            def critic_loss(critic_params, transitions, key):
                sa_encoder_params, g_encoder_params = critic_params["sa_encoder"], critic_params["g_encoder"]
                
                state = transitions.observation
                action = transitions.action
                goal = transitions.extras["goal"]

                sa_repr = sa_encoder.apply(sa_encoder_params, state).reshape(-1, action_size, args.rep_size)
                sa_repr = sa_repr[jnp.arange(sa_repr.shape[0]), action, :]
                g_repr = g_encoder.apply(g_encoder_params, goal)
                
                # InfoNCE
                logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)) #shape = BxB
                
                critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

                # logsumexp regularisation
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

                I = jnp.eye(logits.shape[0])
                correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
                logits_pos = jnp.sum(logits * I) / jnp.sum(I)
                logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

                return critic_loss, (logsumexp, correct, logits_pos, logits_neg)
                
            (loss, (logsumexp, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params, transitions, key)
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
            key, critic_key, actor_key = jax.random.split(key, 3)

            training_state, critic_metrics = update_critic(transitions, training_state, critic_key)

            training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(critic_metrics)
            
            return (training_state, key,), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
                (args.gamma, args.goal_start_channel, args.goal_end_channel), transitions, batch_keys
            )

            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                transitions,
            )
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

            # Fit KDE before collection (using 10k goals)
            kde = stats.gaussian_kde(transitions.extras['goal'][:10000].T, bw_method=0.1)
            training_state = training_state.replace(kde=kde)

            # update buffer
            training_state, env_state, buffer_state = get_experience(
                training_state,
                transitions,
                env_state,
                buffer_state,
                experience_key1,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )

            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]),
                transitions,
            )

            # take actor-step worth of training-step
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

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_training_steps_per_epoch)
            
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics
        
        print('prefilling replay buffer....')
        key, prefill_key = jax.random.split(key, 2)
        training_state, env_state, buffer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        # saving_dir = f"saved/{wandb_group}/{run_name}/"
        saving_dir = f"saved/{args.env_id}/{args.alg_name}_{args.exp_no}/seed_{seed}/"
        os.makedirs(saving_dir, exist_ok=True)
        training_walltime = 0
        print('starting training....')
        for ne in range(args.num_epochs):
            
            t = time.time()
            key, epoch_key = jax.random.split(key)
            training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
                
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / (epoch_training_time + 1e-6)
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            
            metrics, key = run_evaluation(training_state, metrics, key)

            print(metrics)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(np.array(training_state.xy_coverage) / np.sum( np.array(training_state.xy_coverage) ), cmap="viridis", ax=ax)
            # fig.savefig('saved/temp-' + str(ne) + '.png')
            epoch_str = '0' * (3 - len(str(ne))) + str(ne)
            fig.savefig(saving_dir + "epoch" + epoch_str + '.png')

            if args.track:             
                metrics['media/coverage'] = wandb.Image(fig)

                wandb.log(metrics, step=ne)

                if args.wandb_mode == 'offline':
                    trigger_sync()

            plt.close()

        if args.checkpoint:
            # Save current policy and critic params.
            params = (training_state.critic_state.params)
            path = f"{save_path}/final.pkl"
            save_params(path, params)
        
        if args.track:
            wandb.finish()