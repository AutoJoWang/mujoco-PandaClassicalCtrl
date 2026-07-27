import numpy as np
from scipy.spatial.transform import Rotation


class GraspPlanner:

    def __init__(
        self,
        approach_height=0.3
    ):
        self.approach_height = approach_height


    def top_grasp(self, object_pos):

        # 预抓取
        pre_grasp_pos = object_pos.copy()
        pre_grasp_pos[2] += self.approach_height #z


        # 抓取点
        grasp_pos = object_pos.copy()
        grasp_pos[2] += 0.1


        return (
            pre_grasp_pos,
            grasp_pos,
        )