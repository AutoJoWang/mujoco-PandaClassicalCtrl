'''
robot_interface :
    1.封装了mujoco的环境，提供了一个统一的接口
    2.提供了一个统一的接口，方便后续的控制器设计和强化学习算法的实现

'''
import mujoco
import numpy as np


class PandaRobot:

    def __init__(self, env):

        self.env = env
        self.model = env.model
        self.data = env.data

        self.body_ids={}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            self.body_ids[body_name] = i
        
        #reach_task:hands
        self.ee_body_name = "hand"

        self.joint_limits = self.model.jnt_range[:7].copy()

        self.ee_body_id = self.body_ids[self.ee_body_name]


    def get_joint_pos(self):
        """返回7个机械臂关节角（不包含夹爪）"""
        return self.data.qpos[:7].copy()
    
    def get_joint_vel(self):
        return self.data.qvel[:7].copy()
    
    def get_body_pose(self, body_name):

        body_id = self.body_ids[body_name]
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()

        return pos, quat
    #末端执行器的位姿信息
    def get_end_effector_pose(self):

        return self.get_body_pose(self.ee_body_name)
    
    def get_joint_limits(self):
        lower = self.joint_limits[:,0]
        upper = self.joint_limits[:,1]
        return lower,upper

    def set_end_effector(self, body_name):
        self.ee_body_name = body_name
        self.ee_body_id = self.body_ids[body_name]

    def set_joint_target(self, q_target):
        #这里没有使用pid控制器，而是直接将目标位置设置为关节位置
        #因为mxl里的控制器是位置控制器（设置了kp、kd），所以直接设置关节位置即可
        self.data.ctrl[:7] = np.asarray(q_target)


    def set_gripper(self,width):

        self.data.ctrl[7] = width
        # print('gripper ctrl = ',self.data.ctrl[7])

    def gripper_open(self):
        self.set_gripper(255)

    
    def gripper_close(self):
        self.set_gripper(0.0)

    def get_jacp(self):
        '''
        返回末端位置Jacp (3x7)
        '''
        jacp = np.zeros((3,self.model.nv))
        jacr = np.zeros((3,self.model.nv))

        mujoco.mj_jacBody(self.model,self.data,jacp,jacr,self.ee_body_id)

        return jacp[:,:7].copy()

    def get_jacobian(self):
        '''
        返回末端位置Jacp (3x7)
        '''
        jacp = np.zeros((3,self.model.nv))
        jacr = np.zeros((3,self.model.nv))

        mujoco.mj_jacBody(self.model,self.data,jacp,jacr,self.ee_body_id)

        J = np.vstack([jacp,jacr])

        return J[:,:7]
    
    def compute_Jacoobian(self,q):
        '''
        评估一组q时，需要获取Jacobian,但是不想让仿真执行
        所以没法调用仿真读取，于是自己写计算Jacob
        '''
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        self.data.qpos[:7] = q

        mujoco.mj_forward(self.model,self.data)
        J = self.get_jacobian().copy()

        self.restore_state(qpos_backup,qvel_backup)
        return J
     

    
    def restore_state(self, qpos_backup,qvel_backup):
        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)

    def print_robot_info(self):

        print("=" * 50)

        print("Robot 刚体数量")

        print(self.model.nbody)


        for i in range(self.model.nbody):

            print(i,mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i))

        print()

        print("Joint Number")

        print(self.model.njnt)

        for i in range(self.model.njnt):

            print(i,mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))

        print()

        print("Actuator Number")

        print(self.model.nu)

        for i in range(self.model.nu):

            print(i,mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        print()

        print("qpos dimension")

        print(self.model.nq)

        print()

        print("qvel dimension")

        print(self.model.nv)

        print()

        print('site')

        print(self.model.nsite)

        for i in range(self.model.nsite):

            print(i,mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i))

        print("=" * 50)

