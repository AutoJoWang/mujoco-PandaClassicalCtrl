import time
import numpy as np
from simulation.mujoco_env import MujocoEnv
from robot.panda import PandaRobot
from controllers.ik_controller import JacobianIK
from planner.trajectory_generator import TrajectoryGenerator

XML_PATH = "robot_description/franka_emika_panda/scene.xml"


def main():

    env = MujocoEnv(XML_PATH)
    robot = PandaRobot(env)
    ik = JacobianIK()
    planner = TrajectoryGenerator()

    planner.reset()
    control_timer = 0.0
    planner_timer = 0.0

    target_pos = planner.get_target()
    target_pos = np.array([0.2,0.2,0.7])

    with env.launch_viewer() as viewer:

        while viewer.is_running():
            #--------------仿真---------------
            env.step()

            #--------------控制---------------
            control_timer += env.physics_dt
            planner_timer += env.physics_dt

            if control_timer > env.control_dt:
                q_target = ik.pos_solve(robot,target_pos)
                robot.set_joint_target(q_target)

                # print(robot.get_joint_pos())
                # print(np.linalg.norm(target_pos))
                # print(np.linalg.cond(robot.get_jacobian()))

                control_timer = 0.0

            # if planner_timer > env.planner_dt:
            #     planner.update(env.planner_dt)
            #     target_pos = planner.get_target()
            #     env.set_site_position('target',target_pos)

            #     planner_timer = 0.0

            #更新Viewer
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()