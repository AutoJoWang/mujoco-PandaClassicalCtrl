from enum import Enum


class PickPlaceState(Enum):

    INIT = 0

    PRE_GRASP = 1

    GRASP = 2

    CLOSE_GRIPPER = 3

    LIFT = 4

    MOVE_PLACE_PRE = 5

    MOVE_PLACE = 6

    RELEASE = 7

    RETREAT = 8

    DONE = 9