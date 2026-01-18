#!/usr/bin/env python
"""Render trained PQN-CRL policy on Navix environment."""

import jax
import jax.numpy as jnp
import pickle
import imageio
import numpy as np

# Import custom environments BEFORE navix
import mava.wrappers.navix_envs.nx_door_key
import mava.wrappers.navix_envs.nx_four_rooms
import mava.wrappers.navix_envs.nx_empty

import navix as nx
from navix.rendering.cache import render_background, unflatten_patches, TILE_SIZE
from navix.rendering.registry import SPRITES_REGISTRY
from mava.networks.icrl import sa_ConvEncoder_ActionInput, small_G_encoder


def render_state(state, goal_coords=None):
    """Render a Navix state to an RGB image.
    
    Args:
        state: Navix State object
        goal_coords: Optional goal coordinates to highlight
        
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    grid = state.grid
    H, W = grid.shape
    
    # Start with background (floor and walls)
    image = np.array(render_background(grid))
    
    # Place entities on top
    entities = state.entities
    
    # Render player
    if 'player' in entities:
        player = entities['player']
        pos = np.array(player.position).flatten()
        
        if len(pos) >= 2:
            y, x = int(pos[0]), int(pos[1])
            # Only render if position is within grid bounds
            if 0 <= y < H and 0 <= x < W:
                direction = int(np.array(player.direction).flatten()[0]) % 4
                
                # Get player sprite based on direction
                player_sprite = np.array(SPRITES_REGISTRY['player'][direction])
                
                # Place sprite at position
                y_start, x_start = y * TILE_SIZE, x * TILE_SIZE
                y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
                
                # Bounds check
                img_h, img_w = image.shape[:2]
                if y_end <= img_h and x_end <= img_w:
                    # Alpha blend (player sprites might have transparency)
                    mask = np.any(player_sprite > 0, axis=-1, keepdims=True)
                    image[y_start:y_end, x_start:x_end] = np.where(
                        mask, player_sprite, image[y_start:y_end, x_start:x_end]
                    )
    
    # Render door if present
    if 'door' in entities:
        door = entities['door']
        pos = np.array(door.position).flatten()
        
        if len(pos) >= 2:
            y, x = int(pos[0]), int(pos[1])
            # Only render if position is within grid bounds
            if 0 <= y < H and 0 <= x < W:
                colour = int(np.array(door.colour).flatten()[0]) if hasattr(door, 'colour') else 0
                is_open = int(np.array(door.open).flatten()[0]) if hasattr(door, 'open') else 0
                
                # door sprites indexed by [colour, state (closed/open/locked)]
                door_sprite = np.array(SPRITES_REGISTRY['door'][colour, is_open])
                
                y_start, x_start = y * TILE_SIZE, x * TILE_SIZE
                y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
                
                # Bounds check
                img_h, img_w = image.shape[:2]
                if y_end <= img_h and x_end <= img_w:
                    mask = np.any(door_sprite > 0, axis=-1, keepdims=True)
                    image[y_start:y_end, x_start:x_end] = np.where(
                        mask, door_sprite, image[y_start:y_end, x_start:x_end]
                    )
    
    # Render key if present
    if 'key' in entities:
        key = entities['key']
        pos = np.array(key.position).flatten()
        
        # Check if key is valid (not in discard pile position)
        if len(pos) >= 2:
            y, x = int(pos[0]), int(pos[1])
            # Only render if position is within grid bounds
            if 0 <= y < H and 0 <= x < W:
                colour = int(np.array(key.colour).flatten()[0]) if hasattr(key, 'colour') else 0
                
                key_sprite = np.array(SPRITES_REGISTRY['key'][colour])
                
                y_start, x_start = y * TILE_SIZE, x * TILE_SIZE
                y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
                
                # Bounds check
                img_h, img_w = image.shape[:2]
                if y_end <= img_h and x_end <= img_w:
                    mask = np.any(key_sprite > 0, axis=-1, keepdims=True)
                    image[y_start:y_end, x_start:x_end] = np.where(
                        mask, key_sprite, image[y_start:y_end, x_start:x_end]
                    )
    
    # Highlight goal position if provided
    if goal_coords is not None:
        goal_y, goal_x = int(goal_coords[0]), int(goal_coords[1])
        y_start, x_start = goal_y * TILE_SIZE, goal_x * TILE_SIZE
        y_end, x_end = y_start + TILE_SIZE, x_start + TILE_SIZE
        
        # Ensure bounds are valid
        img_h, img_w = image.shape[:2]
        y_end = min(y_end, img_h)
        x_end = min(x_end, img_w)
        
        if y_end > y_start and x_end > x_start:
            # Draw goal marker (green tint)
            if 'goal' in SPRITES_REGISTRY:
                goal_sprite = np.array(SPRITES_REGISTRY['goal'])
                # Resize if needed
                sprite_h, sprite_w = min(y_end - y_start, TILE_SIZE), min(x_end - x_start, TILE_SIZE)
                goal_sprite = goal_sprite[:sprite_h, :sprite_w]
                mask = np.any(goal_sprite > 0, axis=-1, keepdims=True)
                target_slice = image[y_start:y_start+sprite_h, x_start:x_start+sprite_w]
                image[y_start:y_start+sprite_h, x_start:x_start+sprite_w] = np.where(
                    mask, goal_sprite, target_slice
                )
            else:
                # Simple green border for goal
                border = 2
                image[y_start:y_start+border, x_start:x_end] = [0, 255, 0]
                image[y_end-border:y_end, x_start:x_end] = [0, 255, 0]
                image[y_start:y_end, x_start:x_start+border] = [0, 255, 0]
                image[y_start:y_end, x_end-border:x_end] = [0, 255, 0]
    
    return image.astype(np.uint8)


def render_policy(
    env_name: str = "gcrl_door_key",
    checkpoint_path: str = None,
    num_episodes: int = 3,
    max_steps: int = 100,
    temperature: float = 0.1,  # Match training temperature
    output_path: str = "policy_rollout.gif",
    seed: int = 42,
):
    """Render policy rollouts and save as GIF."""
    
    # Get goal coordinates
    if env_name == "gcrl_door_key":
        from mava.wrappers.navix_envs.nx_door_key import door_key_goal_coords as goal_coords
    elif env_name == "gcrl_four_rooms":
        from mava.wrappers.navix_envs.nx_four_rooms import four_rooms_goal_coords as goal_coords
    elif env_name == "gcrl_empty":
        from mava.wrappers.navix_envs.nx_empty import empty_goal_coords as goal_coords
    else:
        raise ValueError(f"Unknown env: {env_name}")
    
    goal_coords_np = np.array(goal_coords)
    
    # Create environment
    env = nx.make(env_name)
    action_size = len(env.action_set)
    rep_size = 64
    grid_shape = (16, 16)
    
    # Initialize networks
    key = jax.random.PRNGKey(seed)
    key, sa_key, g_key = jax.random.split(key, 3)
    
    sa_encoder = sa_ConvEncoder_ActionInput(rep_size=rep_size)
    g_encoder = small_G_encoder(rep_size=rep_size)
    
    # Initialize params (or load from checkpoint)
    dummy_obs = jnp.ones([1, *grid_shape, 1])
    dummy_action = jnp.ones([1, action_size])
    sa_params = sa_encoder.init(sa_key, dummy_obs, dummy_action)
    g_params = g_encoder.init(g_key, jnp.ones([1, 2]))
    
    if checkpoint_path:
        with open(checkpoint_path, 'rb') as f:
            saved_params = pickle.load(f)
        sa_params = saved_params['sa_encoder']
        g_params = saved_params['goal_encoder']
        print(f"Loaded params from {checkpoint_path}")
    else:
        print("WARNING: Using random params! Pass --checkpoint for trained policy.")
        return
    
    # Encode goal
    goal_input = goal_coords.astype(jnp.float32)[None, :]
    g_repr = g_encoder.apply(g_params, goal_input)[0]  # Shape: (rep_size,)
    
    def get_action(obs, key):
        """Get action from policy."""
        batch_obs = obs[None, ...]  # Add batch dim
        
        def compute_q(action_idx):
            a_onehot = jax.nn.one_hot(jnp.array([action_idx]), action_size)
            sa_repr = sa_encoder.apply(sa_params, batch_obs, a_onehot)[0]
            return -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2))
        
        q_values = jax.vmap(compute_q)(jnp.arange(action_size))
        
        # Sample action with temperature (low = near-greedy)
        action = jax.random.categorical(key, q_values / temperature)
        return action, q_values
    
    # JIT compile the action function
    get_action_jit = jax.jit(get_action)
    
    # Debug: print obs shape and first action
    print(f"Goal coords: {goal_coords}")
    print(f"Action size: {action_size}")
    print(f"Grid shape expected: {grid_shape}")
    
    # Collect frames
    all_frames = []
    
    # JIT compile env step
    env_step = jax.jit(env.step)
    
    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")
        key, reset_key, ep_key = jax.random.split(key, 3)
        
        timestep = env.reset(reset_key)
        frames = []
        
        # Debug first step
        if ep == 0:
            print(f"  Observation shape: {timestep.observation.shape}")
            obs = timestep.observation
            ep_key, debug_key = jax.random.split(ep_key)
            _, q_vals = get_action_jit(obs, debug_key)
            print(f"  First Q-values: {np.array(q_vals)}")
            print(f"  Best action: {np.argmax(np.array(q_vals))}")
        
        for step in range(max_steps):
            # Render current state
            frame = render_state(timestep.state, goal_coords_np)
            frames.append(frame)
            
            # Get observation (symbolic)
            obs = timestep.observation
            
            # Get action
            ep_key, action_key = jax.random.split(ep_key)
            action, q_values = get_action_jit(obs, action_key)
            
            # Step environment - pass timestep, not state
            timestep = env_step(timestep, action)
            
            # Check if done (reward > 0 means goal reached)
            if float(timestep.reward) > 0:
                print(f"  Goal reached at step {step + 1}!")
                # Render final frame
                frame = render_state(timestep.state, goal_coords_np)
                frames.append(frame)
                break
        else:
            print(f"  Did not reach goal in {max_steps} steps")
            # Add final frame
            frame = render_state(timestep.state, goal_coords_np)
            frames.append(frame)
        
        all_frames.extend(frames)
        # Add separator between episodes (pause on last frame)
        if ep < num_episodes - 1:
            for _ in range(5):  # Hold last frame for 5 frames
                all_frames.append(frames[-1])
    
    # Save GIF
    print(f"Saving GIF to {output_path}")
    imageio.mimsave(output_path, all_frames, fps=5)
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render trained PQN-CRL policy")
    parser.add_argument("--env", type=str, default="gcrl_door_key",
                        choices=["gcrl_door_key", "gcrl_four_rooms", "gcrl_empty"],
                        help="Environment name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved params pickle file")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to render")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature for action sampling (lower = more greedy)")
    parser.add_argument("--output", type=str, default="policy_rollout.gif",
                        help="Output GIF path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    render_policy(
        env_name=args.env,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        temperature=args.temperature,
        output_path=args.output,
        seed=args.seed,
    )
