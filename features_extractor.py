
import torch.nn as nn
import torch
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class MultiAgentMinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)

        # Get the observation space
        agent_obs_space = observation_space[0]['image']
        n_input_channels = agent_obs_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (5, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )


        with torch.no_grad():
            sample_obs = agent_obs_space.sample()
            n_flatten = self.cnn(torch.as_tensor(sample_obs).float().unsqueeze(0)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:

        image_obs = observation['image'].unsqueeze(0)
        cnn_out = self.cnn(image_obs)

        return self.linear(cnn_out)

