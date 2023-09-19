from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction
from multigrid.core.world_object import Goal, Wall, Door, Key
import numpy as np
class ConstrainedEnv(MultiGridEnv):
           
   def __init__(
         self,
         agent_start_pos = [(1, 1), (5, 3)],
         agent_start_dir = [Direction.right, Direction.right],
         joint_reward: bool = False,
         success_termination_mode: str = 'any',
         max_steps: int = 100,
         **kwargs):
      
      self.agent_start_pos = agent_start_pos
      self.agent_start_dir = agent_start_dir
      self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 50,
            'render_fps': 30,  # Add this line
        }

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
      


