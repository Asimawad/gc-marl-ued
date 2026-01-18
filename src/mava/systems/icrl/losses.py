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

"""
Contrastive loss functions and energy functions for icrl - JaxGCRL: https://github.com/MichalBortkiewicz/JaxGCRL
"""

import jax
import jax.numpy as jnp


def energy_fn(name: str, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute energy (similarity) between representations.

    Args:
        name: Energy function name ("norm", "l2", "dot", "cosine")
        x: State-action representations [batch, repr_dim] or [batch, 1, repr_dim]
        y: Goal representations [batch, repr_dim] or [1, batch, repr_dim]

    Returns:
        Energy values (higher = more similar)
    """
    if name == "norm":
        # Negative L2 norm
        # Range: (-inf, 0], with 0 being identical
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)

    elif name == "l2":
        # Negative squared L2
        # Range: (-inf, 0], with 0 being identical
        return -jnp.sum((x - y) ** 2, axis=-1)

    elif name == "dot":
        # Dot product (used in original InfoNCE paper)
        # Range: (-inf, inf), higher = more similar
        return jnp.sum(x * y, axis=-1)

    elif name == "cosine":
        # Cosine similarity (normalized dot product)
        # Range: [-1, 1], with 1 being identical direction
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True) + 1e-6
        return jnp.sum((x / x_norm) * (y / y_norm), axis=-1)

    else:
        raise ValueError(f"Unknown energy function: {name}. " f"Available: norm, l2, dot, cosine")


def compute_logits(energy_name: str, sa_repr: jnp.ndarray, g_repr: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise logits matrix for contrastive learning.

    Args:
        energy_name: Energy function to use
        sa_repr: State-action representations [batch, repr_dim]
        g_repr: Goal representations [batch, repr_dim]

    Returns:
        Logits matrix [batch, batch] where logits[i,j] = energy(sa_repr[i], g_repr[j])
    """
    # Expand dimensions for broadcasting: [batch, 1, repr_dim] vs [1, batch, repr_dim]
    return energy_fn(energy_name, sa_repr[:, None, :], g_repr[None, :, :])


def contrastive_loss_fn(name: str, logits: jnp.ndarray, logsumexp_penalty_coeff: float = 0.0) -> jnp.ndarray:
    """Compute contrastive loss from logits matrix.

    Args:
        name: Loss function name ("fwd_infonce", "bwd_infonce", "sym_infonce", "binary_nce")
        logits: Pairwise similarity matrix [batch, batch]
        logsumexp_penalty_coeff: Coefficient for logsumexp regularization (default: 0.0)

    Returns:
        Scalar loss value
    """
    if name == "fwd_infonce":
        # Forward InfoNCE: for each (s,a), classify which goal it reaches
        # This is what you currently use
        # Intuition: "Given this state-action, which goal did we actually reach?"
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

    elif name == "bwd_infonce":
        # Backward InfoNCE: for each goal, classify which (s,a) led to it
        # Intuition: "Given this goal, which state-action actually reached it?"
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))

    elif name == "sym_infonce":
        # Symmetric InfoNCE: both directions (more stable, recommended by some papers)
        # Intuition: Learn both "which goal did I reach?" and "which action reached this goal?"
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )

    elif name == "binary_nce":
        # Binary NCE: treat each pair as binary classification
        # Simpler but may be less sample efficient
        I = jnp.eye(logits.shape[0])
        # Positive pairs: sigmoid should be high
        pos_loss = -jnp.mean(jax.nn.log_sigmoid(jnp.diag(logits)))
        # Negative pairs: sigmoid should be low (1 - sigmoid should be high)
        neg_logits = logits * (1 - I)  # Zero out diagonal
        neg_loss = -jnp.mean(jax.nn.log_sigmoid(-neg_logits) * (1 - I))
        critic_loss = pos_loss + neg_loss

    else:
        raise ValueError(
            f"Unknown contrastive loss function: {name}. "
            f"Available: fwd_infonce, bwd_infonce, sym_infonce, binary_nce"
        )

    # Optional: logsumexp regularization (helps with numerical stability)
    if logsumexp_penalty_coeff > 0:
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

    return critic_loss


def compute_contrastive_metrics(logits: jnp.ndarray) -> dict:
    """Compute metrics for monitoring contrastive learning.

    Args:
        logits: Pairwise similarity matrix [batch, batch]

    Returns:
        Dictionary of metrics
    """
    I = jnp.eye(logits.shape[0])

    # Positive vs negative logits
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

    # Classification accuracy
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    categorical_accuracy = jnp.mean(correct)

    # Logsumexp
    logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1).mean()

    return {
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "categorical_accuracy": categorical_accuracy,
        "logsumexp": logsumexp,
    }
