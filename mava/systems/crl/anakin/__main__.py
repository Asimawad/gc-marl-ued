"""
CRL System Entry Point - Direct port of working ICRL implementation

This wraps the original train_icrl_smax.py with Mava's Hydra config system.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from dataclasses import dataclass

# Add current directory to path for local imports
CRL_ANAKIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CRL_ANAKIN_DIR)  # Add this directory to path
sys.path.insert(0, os.path.dirname(CRL_ANAKIN_DIR))  # Add mava/systems/crl to path

# Now we can import from the local module
from mava.systems.crl.anakin.ff_crl import Args, main as crl_main


def config_to_args(config: DictConfig) -> Args:
    """Convert Hydra config to Args dataclass used by original code."""
    
    # Create Args instance with values from config
    args = Args()
    
    # System config
    args.seed = config.system.seed
    args.total_env_steps = config.system.total_env_steps
    args.num_epochs = config.system.num_epochs
    args.num_envs = config.system.num_envs
    args.num_eval_envs = config.system.num_eval_envs
    args.episode_length = config.system.episode_length
    args.unroll_length = config.system.unroll_length
    
    # Learning rates
    args.actor_lr = config.system.actor_lr
    args.critic_lr = config.system.critic_lr
    args.alpha_lr = config.system.alpha_lr
    
    # Buffer
    args.max_replay_size = config.system.max_replay_size
    args.min_replay_size = config.system.min_replay_size
    
    # Training
    args.batch_size = config.system.batch_size
    args.gamma = config.system.gamma
    args.logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    
    # Environment
    args.env_id = "smax"
    args.smax_map_name = config.system.smax_map_name
    args.discrete_actions = config.system.discrete_actions
    
    # Checkpoint
    args.checkpoint = config.system.checkpoint
    
    # Neptune logging
    args.use_neptune = config.system.get("use_neptune", True)
    args.neptune_project = config.system.get("neptune_project", "InstaDeep/ssued")
    args.neptune_tags = config.system.get("neptune_tags", "crl,smax,original")
    
    # Experiment name
    args.exp_name = "crl_smax"
    
    return args


@hydra.main(version_base=None, config_path="../../../configs", config_name="crl_config")
def run_crl(config: DictConfig) -> float:
    """Main entry point for CRL system."""
    

    
    # Convert Hydra config to Args
    args = config_to_args(config)
    
    # Run the original ICRL training logic
    crl_main(args)
    
    return 0.0


if __name__ == "__main__":
    run_crl()

