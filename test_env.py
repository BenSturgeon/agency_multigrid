import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-Custom-v0', agents=1, render_mode='human', autoreset=True)

observations, infos = env.reset()
while not env.is_done():
   # this is where you would insert your policy / policies
   actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()