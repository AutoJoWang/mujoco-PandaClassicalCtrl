import time
import cv2
import numpy as np
from simulation.mujoco_env import MujocoEnv
from robot.panda import PandaRobot
from controllers.ik_controller import JacobianIK
from planner.trajectory_generator import TrajectoryGenerator
from planner.grasp import GraspPlanner
from tasks.pick_place import PickPlaceTask
from camera.camera import Camera
from camera.detector import ColorDetector

XML_PATH = "robot_description/franka_emika_panda/scene.xml"


def main():

    env = MujocoEnv(XML_PATH)
    robot = PandaRobot(env)
    ik = JacobianIK()
    planner_traj = TrajectoryGenerator()
    planner = GraspPlanner()
    task = PickPlaceTask(robot,ik,planner,env.control_dt)
    camera = Camera(env)
    detector = ColorDetector()

    planner_traj.reset()
    control_timer = 0.0
    planner_timer = 0.0
    camera_timer = 0.0
    object_detected = False
    cube_pos = None
    buffer = []

    # object_pos = np.array([-0.2,-0.2,0.2])
    cube_gt = robot.data.body('cube').xpos.copy()
    place_pos = robot.data.body("goal").xpos.copy()

    print('初始角度：',robot.get_joint_pos())


    with env.launch_viewer() as viewer:
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        while viewer.is_running():
            control_timer += env.physics_dt
            
            planner_timer += env.physics_dt
            camera_timer += env.physics_dt
            #--------------仿真---------------
            env.step()
            if not object_detected:
                if camera_timer > 0.05:
                    img = env.render_camera()
                    # cv2.imshow("top_camera",img[:,:,::-1])
                    # cv2.waitKey(1)
                
                    rgb = camera.get_rgb()
                    pixel = detector.detect_red(rgb)
                    if pixel is not None:
                        cube_pos_vision = camera.get_object_world_pos(pixel)
                        if cube_pos_vision is not None:
                            buffer.append(cube_pos_vision)
                            if len(buffer)> 10:
                                buffer.pop(0)
                                cube_pos = np.mean(buffer,axis=0)#0：列平均 1：行平均
                                cube_pos += np.array([0.0,0.1, 0.1])
                                print('obkect pos:',cube_pos)
                                # print(camera.world_to_pixel(cube_gt))
                                task.reset(cube_pos,place_pos)
                                object_detected = True


            #         camera_timer = 0.0
            #-------------debug---------------
                # print("pixel:",pixel)
                # print("vision:",cube_pos_vision)
                # print("ground:",cube_pos_gt)
                # print("error:",np.linalg.norm(cube_pos_vision-cube_pos_gt))
                # camera.debug_pose()
                # camera.debug_world_to_camera(cube_pos_gt)
            # cube_gt = robot.data.body("cube").xpos.copy()

            # pixel_gt = camera.world_to_pixel(cube_gt)

            # depth_img = camera.get_depth()

            # d = depth_img[
            #     int(pixel_gt[1]),
            #     int(pixel_gt[0])
            # ]

            # print(
            #     "gt:",
            #     cube_gt
            # )

            # print(
            #     "pixel:",
            #     pixel_gt
            # )

            # print(
            #     "depth:",
            #     d
            # )
            # point = camera.get_object_world_pos(pixel_gt.astype(int))

            # print('point',point)
            #--------------控制---------------

            if control_timer > env.control_dt:
                task.update()

                control_timer = 0.0

            #更新Viewer
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()