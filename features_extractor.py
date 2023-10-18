
import torch.nn as nn
import torch
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class MultiAgentMinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)

        # Assume all agents have the same observation space
        agent_obs_space = observation_space.spaces[0]['image']
        n_input_channels = agent_obs_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(agent_obs_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
        # Extract image observations for each agent and pass through CNN
        image_obs = torch.stack([obs_dict['image'] for obs_dict in observations.values()])
        cnn_out = self.cnn(image_obs)

        # Pass CNN output through linear layer
        return self.linear(cnn_out)

