from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    STATE_TO_IDX,
    IDX_TO_STATE,
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

from .array import (
    Empty,
    contents,
    can_overlap,
    can_pickup,
    can_contain,
    see_behind,
    toggle,
)
if TYPE_CHECKING:
    from minigrid.minigrid_env import MiniGridEnv

Point = Tuple[int, int]


def world_obj_from_array(array: np.ndarray) -> WorldObj:
    OBJECT_TO_CLS = {
        'wall': Wall,
        'floor': Floor,
        'door': Door,
        'key': Key,
        'ball': Ball,
        'box': Box,
        'goal': Goal,
        'lava': Lava,
    }

    if OBJECT_TO_IDX[array[0]] == 'empty':
        return None
    elif OBJECT_TO_IDX[array[0]] in OBJECT_TO_CLS:
        cls = OBJECT_TO_CLS[IDX_TO_OBJECT[array[0]]]
        return cls.from_array(array)
    raise ValueError(f'Unknown object index: {array[0]}')


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.array = Empty()
        self.array[0] = OBJECT_TO_IDX[type]
        self.array[1] = COLOR_TO_IDX[color]

    @classmethod
    def from_array(cls, array: np.ndarray):
        obj = cls.__new__()
        obj.array = array
        return obj

    @property
    def type(self):
        return IDX_TO_OBJECT[self.array[0]]

    @property
    def color(self):
        return IDX_TO_COLOR[self.array[1]]

    @color.setter
    def color(self, value):
        self.array[1] = COLOR_TO_IDX[value]

    @property
    def contains(self):
        array = contents(self.array)
        if OBJECT_TO_IDX[array[0]] != 'empty':
            return self.__class__

    @contains.setter
    def contains(self, value):
        self.array[4:] = value.array[:4]

    def can_overlap(self) -> bool:
        """Can the agent overlap with this?"""
        return can_overlap(self.array)

    def can_pickup(self) -> bool:
        """Can the agent pick this up?"""
        return can_pickup(self.array)

    def can_contain(self) -> bool:
        """Can this contain another object?"""
        return can_contain(self.array)

    def see_behind(self) -> bool:
        """Can the agent see behind this object?"""
        return see_behind(self.array)

    def toggle(self, env: MiniGridEnv, pos: tuple[int, int]) -> bool:
        """Method to trigger/toggle an action this object performs"""
        original_array = self.array.copy()
        toggle(self.array)
        return np.all(self.array == original_array)

    def encode(self) -> tuple[int, int, int]:
        """Encode the a description of this object as a 3-tuple of integers"""
        return self.array[:3]

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""
        array = Empty()
        array[:3] = type_idx, color_idx, state
        return world_obj_from_array(array)

    def render(self, r: np.ndarray) -> np.ndarray:
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):

    def __init__(self):
        super().__init__('goal', 'green')

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = 'blue'):
        super().__init__('floor', color)

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):

    def __init__(self):
        super().__init__('lava', 'red')

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):

    def __init__(self, color: str = 'grey'):
        super().__init__('wall', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):

    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__('door', color)

    @property
    def is_open(self):
        return STATE_TO_IDX[self.array[2]] == 'open'

    @is_open.setter
    def is_open(self, value):
        if value:
            self.array[2] = STATE_TO_IDX['open']
        elif not self.is_locked:
            self.array[2] = STATE_TO_IDX['closed']

    @property
    def is_locked(self):
        return STATE_TO_IDX[self.array[2]] == 'locked'

    @is_locked.setter
    def is_locked(self, value):
        if value:
            self.array[2] = STATE_TO_IDX['locked']
        elif not self.is_open:
            self.array[2] = STATE_TO_IDX['closed']

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):

    def __init__(self, color: str = 'blue'):
        super().__init__('key', color)

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):

    def __init__(self, color: str = 'blue'):
        super().__init__('ball', color)

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):

    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__('box', color)
        self.contains = contains

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
