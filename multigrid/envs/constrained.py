from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction
from multigrid.core.world_object import Goal, Wall, Door, Key
import numpy as np
from multigrid.core.reward_functions import estimate_entropic_choice_multi_agent


class ConstrainedEnv(MultiGridEnv):
           
   def __init__(
         self,
         algorithm,  
         policy_mapping_fn,  
         agent_start_pos = [(1, 1), (5, 3)],
         agent_start_dir = [Direction.right, Direction.right],
         joint_reward: bool = False,
         success_termination_mode: str = 'any',
         max_steps: int = 200,
         **kwargs):
      
      self.agent_start_pos = agent_start_pos
      self.agent_start_dir = agent_start_dir
      self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50,
            'render_fps': 30,  # Add this line
        }
      
      self.algorithm = algorithm  # Add this line
      self.policy_mapping_fn = policy_mapping_fn  # Add this line

      super().__init__(
      mission_space="get to the green goal square, you rascal",
      
      width = 10,
      height = 5,
      max_steps= max_steps,
      joint_reward=joint_reward,
      success_termination_mode=success_termination_mode,
      **kwargs,
        )
      print(f"{self.max_steps=}")
      self.step_count = 0
      self.agent_start_pos = agent_start_pos
      self.agent_start_dir = agent_start_dir
   
   def _gen_grid(self, width, height):
      """
      :meta private:
      """
      # Create an empty grid
      self.grid = Grid(width, height)

      # Generate the surrounding walls
      self.grid.wall_rect(0, 0, width, height)

      # Place a goal square in the bottom-right corner
      self.put_obj(Goal(), width-2, height-4)

      # Placing barriers
      self.put_obj(Wall(), width-6, height-2)
      self.put_obj(Wall(), width-6, height-4)
      self.put_obj(Wall(), width-3, height-3)
      self.put_obj(Wall(), width-3, height-4)

      self.put_obj(Door('yellow', is_locked=True), 4, 2)
      self.put_obj(Key('yellow'), 6, 1)



      # Place the agent
      for i, agent in enumerate(self.agents):
            if self.agent_start_pos[i] is not None and self.agent_start_dir[i] is not None:
                  agent.state.pos = self.agent_start_pos[i]
                  agent.state.dir = self.agent_start_dir[i]
            else:
                  self.place_agent(agent)

      def render(self, mode="human"):
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        return img
      


      def step(self, actions):
      # Get the action for the second agent
            action = actions[1]

            # Get the current position and direction of the second agent
            agent_pos = self.agents[1].state.pos
            agent_dir = self.agents[1].state.dir

            # Get the position in front of the second agent
            front_pos = agent_pos + Direction.DIR_VEC[agent_dir]

            # Check if the action is to move forward
            if action == 'forward':
                  # Get the object in front of the second agent
                  front_obj = self.grid.get(*front_pos)

                  # Check if the object is a goal
                  if isinstance(front_obj, Goal):
                        # Prevent the second agent from moving onto the goal square
                        actions[1] = 'done'  # Replace the 'forward' action with the 'done' action

            # Call the step method of the superclass to handle the other actions
            obs, rewards, dones, infos = super().step(actions)

            # Check if the second agent has reached the goal
            if isinstance(self.grid.get(*self.agents[1].state.pos), Goal):
                  # Move the second agent back to its previous position
                  self.agents[1].state.pos = agent_pos
                  self.agents[1].state.dir = agent_dir

            return obs, rewards, dones, infos
      
      def _reward(self, i):
            # Get the current agent
            agent = self.agents[i]

            policies = {agent_id: self.algorithm.get_policy(self.policy_mapping_fn(agent_id)) for agent_id in self.agent_ids}

            # Check if the agent is the second agent
            reward = 0
            if i == 1:
                  # Define the reward function for the second agent
                  # For example, give a reward of -1 for each step to encourage the agent to reach the goal as quickly as possible
                  env_copy = copy.deepcopy(self)
                  estimate_entropic_choice_multi_agent(env_copy, policies)
                  reward = 1 - 0.9 * (self.step_count / self.max_steps)
            else:
                  # Define the reward function for the other agents
                  reward = 1 - 0.9 * (self.step_count / self.max_steps)

            return 