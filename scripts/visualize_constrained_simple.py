import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import get_trainable_cls

from train import algorithm_config
from ray.rllib.algorithms import AlgorithmConfig
# Initialize Ray
ray.init()

import argparse
import json

# Define the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="PPO", help="The name of the RLlib-registered algorithm to use.")
parser.add_argument("--framework", type=str, default="torch", help="Deep learning framework to use.")
parser.add_argument("--env", type=str, default="MultiGrid-Constrained-v0", help="MultiGrid environment to use.")
parser.add_argument("--num-agents", type=int, default=2, help="Number of agents in environment.")
parser.add_argument("--num-workers", type=int, default=8, help="Number of rollout workers.")
parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to train on.")
parser.add_argument("--num-timesteps", type=int, default=100, help="Total number of timesteps to visualise.")
parser.add_argument("--load-dir", type=str, default="~/content/ray_results/", help="Directory for saving checkpoints, results, and trained policies.")
args = parser.parse_args()


import datetime
import imageio
import numpy as np

# Define the arguments for the training configuration
parser.add_argument("--lstm", action='store_true', help="Use LSTM model.")
parser.add_argument("--lr", type=float, help="Learning rate for training.")
parser.add_argument("--save-dir", type=str, default='~/ray_results/', help="Directory for saving checkpoints, results, and trained policies.")

# Parse the arguments
args = parser.parse_args()

# Use the arguments to create the config
config = algorithm_config(algo=args.algo, framework=args.framework, env=args.env, num_agents=args.num_agents, num_workers=args.num_workers, num_gpus=args.num_gpus, lstm=args.lstm, lr=args.lr)
checkpoint_path = args.save_dir

# Initialize a list to store frames for gif creation


# Use the arguments to create the config
config = algorithm_config(algo=args.algo, framework=args.framework, env=args.env, num_agents=args.num_agents, num_workers=args.num_workers, num_gpus=args.num_gpus)
checkpoint_path = args.save_dir


# Load the trained agent
checkpoint_path = "~/content/ray_results/"
algo = "PPO"  # The algorithm you used for training
config = {}  # The configuration you used for training
Trainer = get_trainable_cls(algo)
agent = Trainer(config=config)
agent.restore(checkpoint_path)

# Create the environment
env = agent.workers.local_worker().env

# Run the agent in the environment and visualize its behavior
state = env.reset()
done = False

frames = []

while not done:
    action = agent.compute_action(state)
    state, reward, done, info = env.step(action)
    env.render()