import numpy as np


class CartesianTrajectory:


    def __init__(self):

        self.start = None
        self.goal = None

        self.duration = 1.0
        self.time = 0.0

        self.active = False
        self.finished = False



    def generate(self,start,goal,duration):

        self.start = np.array(start)
        self.goal = np.array(goal)

        self.duration = duration

        self.time = 0.0

        self.active = True
        self.finished = False



    def step(self,dt):

        self.time += dt

        s = self.time / self.duration

        if s >= 1.0:
            self.finished = True
            return self.goal

        # smooth interpolation
        s = 3*s*s - 2*s*s*s

        pos = (
            self.start*(1-s)
            +
            self.goal*s
        )

        return pos