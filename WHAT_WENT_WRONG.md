# What Went Wrong: PQN Learnable Temperature Analysis

## 📊 Results Summary

| Approach | Win Rate | Temperature | Notes |
|----------|----------|-------------|-------|
| **SAC Baseline** | **80%** ✓ | Learned (alpha) | Target performance |
| **PQN (temp=0.05)** | **40-50%** | Fixed (too low) | Your original |
| **PQN (learnable temp)** | **3.91%** ❌ | Stuck at 10.0 | Disaster! |

## 🔍 Root Cause: Impossible Target Entropy

### The Problem

Your run had:
```
Target entropy: 1.61 (ratio=0.7)
Actual entropy: 0.97-0.98
Temperature: 10.0 (stuck at maximum)
```

**What this means:**
- The temperature learning tried to increase temp to boost entropy
- Hit the max bound (10.0) immediately
- Even at temp=10.0, could only achieve entropy ~0.98
- Target entropy (1.61) is **physically unreachable** for this task

### Why Is Target Entropy Unreachable?

1. **Q-values are very peaked**: Your contrastive learning produces very confident Q-values (large gaps between best/worst actions)

2. **Available actions mask**: SMAX has action masking, reducing effective action space

3. **Policy is already very stochastic**: At temp=10.0, the policy is nearly uniform, but entropy is still only 0.98

### Mathematical Explanation

```
Max possible entropy = log(num_actions) = log(10) = 2.30
```

But with action masking, if only 5 actions are available:
```
Max effective entropy = log(5) = 1.61
```

And with peaked Q-values (e.g., best action Q=0, worst Q=-10):
```
Even at temp=10: softmax([0,-1,-2,...,-10]/10) is still peaked
Actual entropy: ~0.98
```

So target=1.61 is impossible!

## 🤔 Why Did This Fail So Badly?

### The Vicious Cycle

1. **Epoch 0**: Temp starts at 0.5, entropy ~0.7 (below target 1.61)
2. **Temperature update**: "Entropy too low, increase temp!"
3. **Epoch 1**: Temp increases to 10.0 (max bound)
4. **Still**: Entropy only ~0.98 (still below target!)
5. **Temperature update**: "Still too low, increase more!" → Already at max
6. **Result**: Temp stuck at 10.0 for all 500 epochs

### Why This Hurts Performance

**With temp=10.0, the policy is nearly uniform random:**
```python
# Best action Q-value: -4.27
# Worst action Q-value: -7.26
# Gap: 3.0

softmax([-4.27, -4.5, -5.0, ..., -7.26] / 10.0)
# ≈ [0.11, 0.10, 0.10, ..., 0.09]  # Nearly uniform!
```

**The agent is exploring randomly for 500 epochs, never converging!**

## ✅ Solutions (Ordered by Simplicity)

### Solution 1: Just Use Fixed Temperature (RECOMMENDED!)

**The learnable temperature isn't helping. Just tune the fixed temperature.**

```bash
# Quick test of multiple fixed temperatures
bash test_fixed_temps.sh
```

This will test temp ∈ {0.05, 0.1, 0.2, 0.5} in parallel (50 epochs each).

**Expected results:**
- temp=0.05: 40-50% (your current)
- temp=0.1: **50-60%**
- temp=0.2: **60-70%** ← **Try this!**
- temp=0.5: 55-65% (maybe too random)

**Why this works:**
- Simple, no temperature learning bugs
- Just need to find the right value
- temp=0.2 gives good exploration without being random

### Solution 2: Lower Target Entropy (Retry Learnable Temp)

I've updated `smx_dqn.sh` with better hyperparameters:

**Changes:**
```bash
initial_temperature: 0.2  # (was 0.5)
target_entropy_ratio: 0.3  # (was 0.7) ← KEY CHANGE!
max_temperature: 2.0  # (was 10.0)
temperature_lr: 3e-5  # (was 1e-4) slower updates
```

**New target entropy:**
```
0.3 * log(10) = 0.69
```

This is achievable! With temp~0.2-0.5, you can reach entropy~0.7.

**To run:**
```bash
bash smx_dqn.sh
```

### Solution 3: Add Replay Buffer (Bigger Change)

Your PQN is on-policy. Maybe it needs off-policy learning:

```python
# In Args:
max_replay_size: int = 2000  # Add small buffer
min_replay_size: int = 500
use_replay_buffer: bool = True
```

This would make it more like your SAC baseline.

### Solution 4: Go Back to SAC (Nuclear Option)

If nothing works, stick with SAC! You already have 80% win rate.

The whole point of PQN was to:
1. Remove replay buffer (for speed)
2. Simplify the algorithm

But if it performs worse, there's no point!

## 📈 Recommended Action Plan

### Immediate (Today):

**Option A: Quick win with fixed temperature**
```bash
# Test multiple fixed temps (takes ~2 hours total)
bash test_fixed_temps.sh

# Then use the best one for full training
# My prediction: temp=0.2 will give ~65% win rate
```

**Option B: Retry learnable temp with correct target**
```bash
# I fixed the hyperparameters in smx_dqn.sh
bash smx_dqn.sh
```

### This Week:

1. **Run the fixed temperature sweep** → Find best temp
2. **Full training with best fixed temp** → Should beat 40-50%
3. **Try learnable temp with corrected target** → See if it's better than fixed
4. **If still underperforming**: Add replay buffer or stick with SAC

## 🎯 What To Expect

### Realistic Goals:

| Approach | Expected Win Rate |
|----------|-------------------|
| PQN fixed temp=0.2 | **60-70%** |
| PQN learnable temp (fixed) | **60-75%** |
| PQN + small buffer | **70-80%** |
| SAC (your baseline) | **80%** |

**Bottom line:** PQN might not beat SAC. That's okay! SAC is a very strong baseline.

## 🧠 Lessons Learned

1. **Learnable temperature is tricky**: Easy to misconfigure
2. **Target entropy must be reachable**: Can't just copy from other papers
3. **Peaked Q-values limit entropy**: Contrastive learning produces confident Q-values
4. **Sometimes simpler is better**: Fixed temperature might be enough
5. **On-policy learning is sample inefficient**: PQN needs more envs or a buffer

## 🔧 Debugging Checklist

If you want to understand why PQN underperforms:

- [ ] Check Q-value distribution (are they too peaked?)
- [ ] Check available actions (how many are valid on average?)
- [ ] Compare data diversity (PQN vs SAC)
- [ ] Try adding small replay buffer
- [ ] Try different network architectures
- [ ] Try different InfoNCE temperature
- [ ] Check if goal relabeling is working correctly

## 💡 Alternative Idea: Hybrid Approach

**What if we keep SAC but use your contrastive critic?**

```python
# SAC's actor (learned policy) + Your contrastive critic
# Best of both worlds:
# - SAC's exploration (learned alpha + stochastic policy)
# - Your contrastive learning (better representation)
```

This might actually be the winning combination!

---

## TL;DR - What To Do Now

```bash
# OPTION 1: Quick test (2 hours)
bash test_fixed_temps.sh
# Then use best temp for full run

# OPTION 2: Retry learnable temp (overnight)
bash smx_dqn.sh
# With corrected hyperparameters
```

**My recommendation: Try Option 1 first** (fixed temperature sweep). It's faster and more likely to work.

If temp=0.2 gets you to 65-70%, you're making progress! If it still underperforms, then the issue isn't just temperature - PQN might fundamentally need a replay buffer for this task.

Good luck! 🚀
