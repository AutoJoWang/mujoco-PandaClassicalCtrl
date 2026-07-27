import numpy as np
from tasks.state import PickPlaceState
from planner.trajectory import CartesianTrajectory

class PickPlaceTask:
    def __init__(
        self,
        robot,
        ik,
        planner,
        dt
    ):
        self.robot = robot
        self.ik = ik
        self.planner = planner
        self.dt = dt
        self.state = PickPlaceState.INIT
        self.traj = CartesianTrajectory()

        self.pre_pos = None
        self.grasp_pos = None
        self.place_pos = None
        self.place_pre_pos = None
        self.cnt = 0


    def reset(self,cube_pos,place_pos):
        self.pre_pos, self.grasp_pos = self.planner.top_grasp(cube_pos)
        self.place_pos = place_pos + np.array([0,0,0.1])
        self.place_pre_pos = place_pos + np.array([0,0,0.3])
        self.state = PickPlaceState.PRE_GRASP


    def update(self):

        if self.state == PickPlaceState.PRE_GRASP:
            
            if self.execute_move(self.pre_pos,1.0):
                print("Reach Pre Grasp")
                self.state = PickPlaceState.GRASP

        elif self.state == PickPlaceState.GRASP:

            if self.execute_move(self.grasp_pos,1.0):

                print("Reach Grasp")
                self.state = PickPlaceState.CLOSE_GRIPPER


        elif self.state == PickPlaceState.CLOSE_GRIPPER:
            self.robot.gripper_close()
            self.cnt +=1
            if self.cnt > 30:
                self.state = PickPlaceState.LIFT
                print("CLOSE")
                self.cnt = 0


        elif self.state == PickPlaceState.LIFT:

            if self.execute_move(self.pre_pos,1.0):
                print("Lift done")
                self.state = PickPlaceState.MOVE_PLACE_PRE

        elif self.state == PickPlaceState.MOVE_PLACE_PRE:

            if self.execute_move(self.place_pre_pos,1.5):
                print('Arrive pre place')
                self.state = PickPlaceState.MOVE_PLACE

        elif self.state == PickPlaceState.MOVE_PLACE:
            if self.execute_move(self.place_pos,1.0):
                print('Arrive place')
                self.state = PickPlaceState.RELEASE

        elif self.state == PickPlaceState.RELEASE:

            print("OPEN GRISPER")
            self.robot.gripper_open()
            self.state = PickPlaceState.RETREAT

        elif self.state == PickPlaceState.RETREAT:

            if self.execute_move(self.place_pre_pos,1.0):
                print('Place finished')
                self.state = PickPlaceState.DONE


    def move(self,target):

        q_target = self.ik.solve(self.robot,target)

        self.robot.set_joint_target(q_target)


    def start_trajectory(self,target,duration):

        current,_ = self.robot.get_end_effector_pose()
        self.traj.generate(current,target,duration)


    def execute_move(self,target,duration):

        if not self.traj.active :
            self.start_trajectory(target,duration)

        target_pos=self.traj.step(self.dt)

        self.move(target_pos)

        if self.reached(target):
            self.traj.active = False
            self.traj.finished = False
            return True

        return False
    

    def reached(self,target):

        current,_ = self.robot.get_end_effector_pose()
        pos_error=np.linalg.norm(target-current) < 0.03

        return pos_error