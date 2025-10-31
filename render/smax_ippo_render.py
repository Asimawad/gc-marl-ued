import jax
import pickle
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jaxmarl.viz.visualizer import SMAXVisualizer
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax
from envs.smax import SmaxEnv
import tyro
from etils import epath
from typing import Sequence, Dict
import time

from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

# Render parameters
RANDOM_SEED = 1 # set to int(time.time()) for truly random
MAX_STEPS = 100

# Learned policy network (IPPO No RNN)
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs = x
        
        # first layer kept as before
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        # rest remains the same
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
#        unavail_actions = 1 - avail_actions
        action_logits = actor_mean # - (unavail_actions * 1e10)
        
#        from scipy.special import softmax
#        jax.debug.print("action probs: {}", softmax(action_logits, axis=-1))

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

# Load parameters
def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)
    
# Initialize environment and network
scenario = map_name_to_scenario('2s3z')
env = HeuristicEnemySMAX(scenario=scenario, 
                see_enemy_actions = True, walls_cause_death = True, attack_mode = "closest",
                action_type = 'discrete')
actor = ActorCritic(action_dim=env.action_space(env.agents[0]).n, config={'FC_DIM_SIZE': 128, 'GRU_HIDDEN_DIM': 128})
actor_params = tyro.cli(load_params)
# actor_params = load_params(PARAM_PATH)

state_seq = []
key = jax.random.PRNGKey(RANDOM_SEED)
key_r, key = jax.random.split(key, 2)
obs, state = env.reset(key_r)
max_steps = MAX_STEPS

for _ in range(max_steps):
    # Iterate random keys and sample actions
    key, key_sample, key_step = jax.random.split(key, 3)
    action, _ = actor.apply(actor_params, jnp.stack([obs[a] for a in env.agents]))
    action = action.sample(seed=key_sample)
    action = {a: action[i] for i,a in enumerate(env.agents)}

    state_seq.append((key_sample, state, action))
#    print('action: ', action)
    enemy_healths = state.state.unit_health[env.num_agents:env.num_agents+env.num_enemies]
    max_healths = env.unit_type_health[state.state.unit_types[jnp.arange(env.num_enemies)+env.num_agents]]
#    enemy_healths = enemy_healths / max_healths

    print('enemy healths: ', enemy_healths)

    # Step environment
    obs, state, _,_,_ = env.step(key_step, state, action)

viz = SMAXVisualizer(env, state_seq)
viz.animate(view=True)
