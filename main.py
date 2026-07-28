import time
import mujoco
import cv2
import numpy as np
from simulation.mujoco_env import MujocoEnv
from robot.panda import PandaRobot
from controllers.ik_controller import JacobianIK
from planner.trajectory_generator import TrajectoryGenerator
from planner.grasp import GraspPlanner
from tasks.pick_place import PickPlaceTask


XML_PATH = "robot_description/franka_emika_panda/scene.xml"


def main():

    env = MujocoEnv(XML_PATH)
    robot = PandaRobot(env)
    ik = JacobianIK()
    planner_traj = TrajectoryGenerator()
    planner = GraspPlanner()
    task = PickPlaceTask(robot,ik,planner,env.control_dt)

    planner_traj.reset()
    control_timer = 0.0
    planner_timer = 0.0
    camera_timer = 0.0

    object_pos = np.array([-0.2,-0.2,0.2])
    cube_pos = robot.data.body("cube").xpos.copy()
    place_pos = robot.data.body("goal").xpos.copy()
    task.reset(cube_pos,place_pos)

    # print('初始角度：',robot.get_joint_pos())


    with env.launch_viewer() as viewer:
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        while viewer.is_running():
            control_timer += env.physics_dt
            
            planner_timer += env.physics_dt
            #--------------仿真---------------
            env.step()
            
            #--------------控制---------------

            if control_timer > env.control_dt:
                task.update()

                control_timer = 0.0

            #更新Viewer
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()