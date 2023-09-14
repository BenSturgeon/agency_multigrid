import gymnasium as gym
import multigrid.envs

env = gym.make('MultiGrid-Empty-8x8-v0', agents=2, render_mode='human')

observations, infos = env.reset()
while not env.is_done():
   # this is where you would insert your policy / policies
   actions = {agent.index: agent.action_space.sample() for agent in env.agents}
   observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()