import numpy as np


class TrajectoryGenerator:

    def __init__(self):
        self.time = 0.0

        # 当前目标
        self.target = np.array([
            0.4,
            0.4,
            0.8
        ])

    def reset(self):
        self.time = 0.0

    def update(self, dt):
        """
        每个控制周期调用一次
        """
        self.time += dt

        r =0.05
        self.target = np.array([
            0.4 + r*np.cos(self.time),
            0.4 + r*np.sin(self.time),
            0.8
        ])



    def get_target(self):
        return self.target.copy()