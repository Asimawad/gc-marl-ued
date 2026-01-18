# PQN Improvement Plan: From 40-50% to 80%+ Win Rate

## Current Status
- **SAC Baseline (train_icrl_smax.py)**: 80%+ win rate ✓
- **Your PQN (train_pqn_smax.py)**: 40-50% win rate ❌
- **Goal**: Make PQN outperform SAC baseline

## Root Cause Analysis: Why is PQN Underperforming?

### 1. **Exploration Problem** (CRITICAL)
```python
# Your PQN: Fixed temperature = 0.05 (very low!)
temperature: float = 0.05
```
- Temperature of 0.05 is EXTREMELY low → nearly greedy policy
- SAC has learned alpha that adapts based on entropy
- Low exploration early in training = poor experience diversity

**Evidence from CRL v2 logs:**
- Policy entropy: 0.37-0.75
- Target entropy: 2.07
- The policy is WAY too deterministic!

### 2. **Sample Efficiency Problem**
**PQN (on-policy):**
- Collects 512 envs × 100 steps = 51,200 steps
- Uses each transition ONCE
- Throws away all experience after 1 gradient update

**SAC (off-policy):**
- Stores 5000 trajectories in replay buffer
- Samples and reuses transitions many times
- Better sample efficiency

### 3. **Action Selection Mechanism**
**PQN:**
```python
actions = jax.random.categorical(key, Q_values / temperature)
```
- Samples from Q-values directly
- No explicit policy network
- Temperature is the ONLY exploration mechanism

**SAC:**
```python
mean, log_std = actor.apply(params, obs)
actions = tanh(mean + std * gumbel_noise)
```
- Explicit learned policy with stochasticity
- Gumbel noise for discrete actions
- Entropy bonus in actor loss

### 4. **Training Dynamics**
**Your PQN batch size confusion:**
```python
batch_size: int = 256  # Comment says "InfoNCE needs small batches"
# But actual batch collected = 512 * 5 * 100 = 256,000 transitions!
```
- You're collecting MASSIVE amounts of data per epoch
- Then breaking into mini-batches of 256
- This might be too aggressive for on-policy learning

## Improvement Roadmap (Try in Order)

### PHASE 1: Fix Exploration (Highest Priority!)

#### Experiment 1A: Use Learnable Temperature
```bash
python train_pqn_smax_learnable_temp.py \
  --smax_map_name smacv2_5_units \
  --target_entropy_ratio 0.6 \
  --initial_temperature 0.5 \
  --seed 1
```

**Expected improvement: +10-20% win rate**

#### Experiment 1B: Try Higher Fixed Temperatures
Test multiple values to find the sweet spot:
```bash
# High exploration
python train_pqn_smax.py --temperature 0.5 --seed 1

# Medium exploration
python train_pqn_smax.py --temperature 0.2 --seed 2

# Current (too low)
python train_pqn_smax.py --temperature 0.05 --seed 3
```

**Expected: 0.2-0.5 should perform better than 0.05**

#### Experiment 1C: Add Entropy Bonus to Critic Loss
```python
def critic_loss(critic_params, transitions, key):
    # ... existing code ...
    critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

    # ADD: Encourage high entropy early in training
    q_values = compute_q_values(critic_params, obs, goal)
    policy = jax.nn.softmax(q_values / temperature, axis=-1)
    entropy = -jnp.sum(policy * jnp.log(policy + 1e-8), axis=-1)
    entropy_bonus = 0.01  # Tune this
    critic_loss -= entropy_bonus * jnp.mean(entropy)

    return critic_loss
```

### PHASE 2: Improve Sample Efficiency

#### Experiment 2A: Add a Small Replay Buffer
Even a small buffer can help:
```python
# In Args:
max_replay_size: int = 2000  # Small buffer (vs 5000 in SAC)
min_replay_size: int = 500
use_replay_buffer: bool = True  # NEW FLAG

# Modify training to use buffer when enabled
if args.use_replay_buffer:
    # Store experience in buffer
    # Sample from buffer for training
else:
    # Current PQN approach (train on fresh data)
```

**Expected improvement: +5-10% win rate**

#### Experiment 2B: Increase Num Envs for More Diversity
```python
num_envs: int = 1024  # Double from 512
```
More parallel envs = more diverse on-policy experience

#### Experiment 2C: Shorter Unrolls, More Frequent Updates
```python
unroll_length: int = 50  # Half of current 100
# This doubles the number of training steps per epoch
# More frequent updates might help on-policy learning
```

### PHASE 3: Match SAC's Action Selection

#### Experiment 3A: Add Gumbel Noise (like SAC)
```python
def actor_step(...):
    logits = compute_q_values(action_params, state, goal)
    masked_logits = logits - ((1 - avail_actions) * 1e10)

    # CHANGE: Add Gumbel noise like SAC
    gumbel_noise = jax.random.gumbel(key, shape=masked_logits.shape)
    actions = jnp.argmax((masked_logits / temperature) + gumbel_noise, axis=-1)
```

This is more similar to SAC's exploration mechanism.

#### Experiment 3B: Two-Stage Action Selection
```python
# Stage 1: Sample multiple actions
num_action_samples = 5
keys = jax.random.split(key, num_action_samples)
action_samples = jax.vmap(lambda k: jax.random.categorical(
    k, masked_logits / temperature
))(keys)

# Stage 2: Choose best Q-value among samples
q_vals = jax.vmap(lambda a: compute_q_values(...)[a])(action_samples)
best_idx = jnp.argmax(q_vals)
action = action_samples[best_idx]
```

### PHASE 4: Training Hyperparameters

#### Experiment 4A: Tune Learning Rates
```python
# Try higher critic LR (you're at 1e-4, SAC uses 3e-4)
critic_lr: float = 3e-4

# Try scheduled LR decay
tx = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adam(learning_rate=optax.cosine_decay_schedule(
        init_value=3e-4,
        decay_steps=args.total_env_steps,
        alpha=0.1  # Final LR = 0.1 * initial
    )),
)
```

#### Experiment 4B: Tune Batch Size and Update Frequency
Current effective batch size per gradient step is:
```
num_envs_agents * unroll_length / num_mini_batches
= (512 * 5 * 100) / (256,000 / 256)
= 256,000 / 1000 = 256 ✓
```

Try:
```python
batch_size: int = 512  # Larger mini-batches
# OR
batch_size: int = 128  # Smaller mini-batches, more updates
```

#### Experiment 4C: Stronger Target Network Updates
```python
target_tau: float = 0.005  # Faster target updates (from 0.001)
# OR try hard updates every N steps
use_hard_target_updates: bool = True
target_update_interval: int = 100
```

### PHASE 5: Architecture Changes

#### Experiment 5A: Increase Representation Size
```python
rep_size: int = 128  # Double from 64
```

Bigger representations might capture more complex state-action relationships.

#### Experiment 5B: Add Dropout for Regularization
```python
class SA_encoder(nn.Module):
    @nn.compact
    def __call__(self, s, a, training=True):
        x = jnp.concatenate([s, a], axis=-1)
        x = nn.Dense(1024)(x)
        x = nn.LayerNorm()(x)
        x = nn.swish(x)
        x = nn.Dropout(rate=0.1, deterministic=not training)(x)  # ADD
        # ... rest of network
```

#### Experiment 5C: Separate Encoders for Each Agent
```python
# If you have 5 agents, create 5 separate critics
# This allows specialization but requires more memory
```

### PHASE 6: Curriculum Learning

#### Experiment 6A: Warm-Start with Higher Temperature
```python
# Start with high exploration, gradually reduce
def get_temperature_schedule(step, total_steps):
    progress = step / total_steps
    temp_start = 1.0
    temp_end = 0.1
    return temp_start * (1 - progress) + temp_end * progress
```

#### Experiment 6B: Progressive Difficulty
```python
# Start on easier maps, gradually increase difficulty
# 2s3z → 3s5z → smacv2_5_units → smacv2_10_units
```

## Debugging Checklist

Before trying experiments, verify these aren't broken:

- [ ] **Goal relabeling is working correctly**
  ```python
  # Add debug prints to check future_state
  jax.debug.print("Original goal: {}, Relabeled goal: {}",
                  obs[:, goal_start_idx:goal_end_idx],
                  future_state[:, goal_start_idx:goal_end_idx])
  ```

- [ ] **Available actions mask is applied correctly**
  ```python
  # Verify invalid actions get -1e10
  jax.debug.print("Avail mask: {}, Logits: {}", avail_actions[0], logits[0])
  ```

- [ ] **Batch processing is correct**
  ```python
  # Check shapes throughout the pipeline
  jax.debug.print("Transitions shape: {}", transitions.observation.shape)
  ```

- [ ] **Q-values are in reasonable range**
  ```python
  # Q-values should be negative (distance-based)
  jax.debug.print("Q-value range: [{}, {}]",
                  jnp.min(q_values), jnp.max(q_values))
  ```

## Recommended Experiment Order

### Week 1: Exploration Fixes
1. **Day 1-2**: Run `train_pqn_smax_learnable_temp.py` ← **Start here!**
2. **Day 3-4**: Try fixed temperatures [0.1, 0.2, 0.5, 1.0]
3. **Day 5**: Add entropy bonus to critic loss

**Expected outcome: Should reach 55-65% win rate**

### Week 2: Sample Efficiency
4. **Day 1-2**: Add small replay buffer (size=2000)
5. **Day 3-4**: Increase num_envs to 1024
6. **Day 5**: Try shorter unroll_length=50

**Expected outcome: Should reach 65-75% win rate**

### Week 3: Action Selection & Fine-tuning
7. **Day 1-2**: Add Gumbel noise to action selection
8. **Day 3**: Tune learning rates
9. **Day 4**: Tune batch sizes
10. **Day 5**: Try larger rep_size=128

**Expected outcome: Should reach 75-85% win rate** ✓ **BEATS SAC!**

## Quick Wins (Try First!)

### Experiment ZERO: Sanity Check
```bash
# Run SAC baseline to confirm 80% is reproducible
python train_icrl_smax.py \
  --smax_map_name smacv2_5_units \
  --seed 1 \
  --track

# Then run PQN with learnable temp
python train_pqn_smax_learnable_temp.py \
  --smax_map_name smacv2_5_units \
  --seed 1 \
  --track
```

### Experiment ONE: Copy SAC's Exploration
**Modify your PQN to exactly match SAC's temperature:**
1. Initial temperature: 0.3 (not 0.05!)
2. Learnable with entropy target
3. Target entropy ratio: 0.8 (like SAC)

**This single change might give you +20% win rate!**

### Experiment TWO: Match SAC's Hyperparameters
```python
# Change these in train_pqn_smax.py to match SAC:
critic_lr: float = 3e-4  # SAC's value (not 1e-4)
batch_size: int = 256    # Same as SAC ✓
gamma: float = 0.99      # Same as SAC ✓
logsumexp_penalty_coeff: float = 0.1  # Same as SAC ✓
```

## Key Insights

1. **PQN's main weakness vs SAC**: Lack of exploration diversity
   - SAC has learned alpha + actor stochasticity
   - Your PQN only has temperature=0.05

2. **PQN's main strength**: No replay buffer overhead
   - Should be faster (more SPS)
   - Simpler code
   - But needs MORE parallel envs to compensate

3. **The irony**: You made PQN too "on-policy"
   - On-policy algorithms need HIGH exploration
   - But you set temperature=0.05 (nearly deterministic)
   - This is the worst of both worlds!

## Success Metrics

Track these metrics to understand what's working:

```python
metrics = {
    "exploration/temperature": temperature,
    "exploration/policy_entropy": policy_entropy,
    "exploration/entropy_ratio": policy_entropy / max_entropy,
    "training/q_value_mean": jnp.mean(q_values),
    "training/q_value_std": jnp.std(q_values),
    "training/categorical_accuracy": accuracy,
    "eval/win_rate": win_rate,
    "eval/episode_return": episode_return,
}
```

Good signs:
- Policy entropy starting high (>1.5) and gradually decreasing
- Temperature adapting to maintain target entropy
- Categorical accuracy increasing over time
- Win rate showing steady improvement

Bad signs:
- Entropy collapsing to 0 early
- Temperature stuck at min or max
- Categorical accuracy plateauing below 10%
- Win rate stuck below 30%

## Nuclear Option: Hybrid PQN-SAC

If nothing else works, create a hybrid:
- Keep SAC's actor network and exploration
- Replace SAC's critic with your contrastive critic
- Keep the replay buffer

This would be "SAC with contrastive learning" rather than pure PQN, but might perform best.

---

## TL;DR - Do This First

```bash
# 1. Fix the temperature (this is probably 80% of the problem!)
python train_pqn_smax_learnable_temp.py \
  --smax_map_name smacv2_5_units \
  --temperature_lr 1e-4 \
  --initial_temperature 0.5 \
  --target_entropy_ratio 0.7 \
  --num_envs 1024 \
  --critic_lr 3e-4 \
  --seed 1 \
  --track

# 2. If that doesn't work, add a small replay buffer
# 3. If that doesn't work, increase exploration further
# 4. If that doesn't work, try the hybrid approach
```

**Prediction: Just fixing the temperature will get you to 60-70% win rate!**

Good luck! 🚀
