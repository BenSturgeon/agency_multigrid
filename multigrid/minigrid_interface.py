import numpy as np

from typing import Sequence

from .core.world_object import WorldObj
from .multigrid_env import MultiGridEnv



class MiniGridInterface(MultiGridEnv):
    """
    MultiGridEnv interface for compatibility with single-agent MiniGrid environments.

    Most environment implementations deriving from `minigrid.MiniGridEnv` can be
    converted to a single-agent `MultiGridEnv` by simply inheriting from
    `MiniGridInterface` instead (along with using the multigrid grid and grid objects).

    Examples
    --------
    Start with a single-agent minigrid environment:

    >>> from minigrid.core.world_object import Ball, Key, Door
    >>> from minigrid.core.grid import Grid
    >>> from minigrid.minigrid_env import MiniGridEnv

    >>> class MyEnv(MiniGridEnv):
    >>>    ... # existing class definition

    Now use multigrid imports, keeping the environment class definition the same:

    >>> from multigrid.core.world_object import Ball, Key, Door
    >>> from multigrid.core.grid import Grid
    >>> from multigrid.minigrid_interface import MiniGridInterface as MiniGridEnv

    >>> class MyEnv(MiniGridEnv):
    >>>    ... # same class definition
    """

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        return (item[0] for item in result)

    def step(self, action: int):
        result = super().step({0: action})
        return (item[0] for item in result)

    @property
    def agent_pos(self) -> np.ndarray[int]:
        """
        Get agent position.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Try `env.agents[i].pos` instead."
        )
        return self.agents[0].pos

    @agent_pos.setter
    def agent_pos(self, value: Sequence[int]):
        """
        Set agent position.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Try `env.agents[i].pos` instead."
        )
        if value:
            self.agents[0].pos = value

    @property
    def agent_dir(self) -> int:
        """
        Get agent direction.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Try `env.agents[i].dir` instead."
        )
        return self.agents[0].dir

    @agent_dir.setter
    def agent_dir(self, value: Sequence[int]):
        """
        Set agent direction.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Try `env.agents[i].dir` instead."
        )
        self.agents[0].dir = value

    @property
    def carrying(self) -> WorldObj:
        """
        Get object carried by agent.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Try `env.agents[i].carrying` instead."
        )
        return self.agents[0].carrying

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent.
        """
        assert len(self.agents) == 1, (
           "This property is not supported for multi-agent envs. "
           "Try `env.agents[i].front_pos` instead."
        )
        return self.agents[0].front_pos
