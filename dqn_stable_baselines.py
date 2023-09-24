import multigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
from features_extractor import MinigridFeaturesExtractor
from multigrid.envs.empty import EmptyEnv
import gymnasium as gym

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = EmptyEnv(size=8)
env = ImgObsWrapper(env)

model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(2e5)