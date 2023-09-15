import gymnasium as gym
import multigrid.envs
import time

env = gym.make('MultiGrid-Custom-v0', agents=1, render_mode='human', autoreset=True)
delay_seconds = 0.5

observations, infos = env.reset()
while not env.is_done():
   # this is where you would insert your policy / policies
   actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)
   time.sleep(delay_seconds)

env.close()