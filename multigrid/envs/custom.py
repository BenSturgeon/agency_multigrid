from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction
from multigrid.core.world_object import Goal, Wall

class CustomEnv(MultiGridEnv):
   def __init__(
         self,
         agent_start_pos = (1, 1),
         agent_start_dir = Direction.right,
         joint_reward: bool = False,
         success_termination_mode: str = 'any',
         **kwargs):
      
      self.agent_start_pos = agent_start_pos
      self.agent_start_dir = agent_start_dir

      super().__init__(
      mission_space="get to the green goal square, you rascal",
      width = 10,
      height = 5,
      max_steps= 20,
      joint_reward=joint_reward,
      success_termination_mode=success_termination_mode,
      **kwargs,
        )
   
   def _gen_grid(self, width, height):
      """
      :meta private:
      """
      # Create an empty grid
      self.grid = Grid(width, height)

      # Generate the surrounding walls
      self.grid.wall_rect(0, 0, width, height)

      # Place a goal square in the bottom-right corner
      self.put_obj(Goal(), width-4, height-3)

      # Placing barriers
      self.put_obj(Wall(), width-6, height-2)
      self.put_obj(Wall(), width-6, height-4)
      self.put_obj(Wall(), width-3, height-3)
      self.put_obj(Wall(), width-3, height-4)


      # Place the agent
      for agent in self.agents:
         if self.agent_start_pos is not None and self.agent_start_dir is not None:
               agent.state.pos = self.agent_start_pos
               agent.state.dir = self.agent_start_dir
         else:
               self.place_agent(agent)