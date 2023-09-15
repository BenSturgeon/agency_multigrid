import gymnasium as gym
import multigrid.envs
import time
from imageio import mimsave
from PPO import MinigridFeaturesExtractor
from stable_baselines3 import PPO
from minigrid.wrappers import ImgObsWrapper

def main():
   policy_kwargs = dict(
      features_extractor_class=MinigridFeaturesExtractor,
      features_extractor_kwargs=dict(features_dim=128),
      )
   env = gym.make('MultiGrid-Custom-v0', agents=1, render_mode='rgb_array', autoreset=True)
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

if __name__ == "__main__":
   main()
