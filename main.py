import gymnasium as gym
import multigrid.envs
import time
import nbimporter
from imageio import mimsave
from PPO import MinigridFeaturesExtractor
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper


   policy_kwargs = dict(
   features_extractor_class=MinigridFeaturesExtractor,
   features_extractor_kwargs=dict(features_dim=128),
   )
env = gym.make('MultiGrid-Empty-6x6-v0', render_mode='rgb_array', agents =2 )
env = ImgObsWrapper(env)

delay_seconds = 0.5
images = []

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(2e5)  

observations, infos = env.reset()
step = 0

print("running")
while not env.is_done():
   # Render the environment and append the resulting image to the list
   images.append(env.render('rgb_array'))

   actions = {agent.index: model.predict(observations[agent.index])[0] for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)
   time.sleep(delay_seconds)
   step += 1



mimsave('simulation.gif', images)
env.close()


