
import torch.nn as nn
import torch
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        print(observation_space)
        
        n = observation_space.shape[0]
        m = observation_space.shape[1]
        print(n,m)
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64


        # Compute shape by doing one forward pass
        tens= torch.as_tensor(observation_space.sample()[None]).float()
        tens = torch.tensor(tens).float().permute(0,3,1,2)
        with torch.no_grad():
            n_flatten = self.cnn(tens).shape[1]
        lin = nn.Linear(n_flatten, features_dim)
        self.linear = nn.Sequential(lin, nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = torch.Tensor(observations)
        return self.linear(self.cnn(observations))