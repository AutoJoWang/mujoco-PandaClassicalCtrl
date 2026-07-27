import numpy as np
from scipy.spatial.transform import Rotation

class JacobianIK:

    def __init__(self,step_size=0.3,damping=0.05):
        self.step_size = step_size
        self.damping = damping
        self.position_tolerance = 0.002
        self.rotation_tolerance = np.deg2rad(1)
        self.cnt =0 
    
    def pos_solve(self, robot, target_pos):
        '''
        仅实现机械臂hand到达目标点，没有特定姿势
        输入：关节角、世界坐标位置、目标坐标位置，雅可比矩阵
        输出：关节角度
        '''
        max_dq = 1.0      # rad    
        lower_limit, upper_limit  = robot.get_joint_limits()
        
        current_q = robot.get_joint_pos()  #关节角
        current_pos, _ = robot.get_end_effector_pose()  #世界坐标 FK获得
        J = robot.get_jacp()
        error = target_pos - current_pos

        #达到目标点后不用重复计算，直接返回
        if np.linalg.norm(error) < self.position_tolerance:
            return robot.get_joint_pos()

        # dq = np.linalg.pinv(J) @ error
        #防止奇异位
        JT = J.T
        lambda2 = self.damping**2
        dq = JT @ np.linalg.solve(
            J @ JT + lambda2 * np.eye(3),
            error
        )

        dq = np.clip(dq,-max_dq,max_dq)

        q_target = current_q + dq*self.step_size

        q_target = np.clip(q_target,lower_limit,upper_limit)

        return q_target
    
    def solve(self,robot,target_pos):
        '''
        target_pos:x,y,z位置
        rpy:欧拉角
        '''
        max_dq = 1.0     # rad
        current_pos, current_quat = robot.get_end_effector_pose()  #世界坐标 FK获得
        J = robot.get_jacobian()
        current_q = robot.get_joint_pos()
        lower_limit, upper_limit  = robot.get_joint_limits()

        object_normal = np.array([0,0,1])

        target_rot = self.compute_grasp_rotation(object_normal)
        current_quat = np.array([current_quat[1],current_quat[2],current_quat[3],current_quat[0]]) #(w,x,y,z)-->(x,y,z,w)
        current_rot = Rotation.from_quat(current_quat).as_matrix()

        pos_error = target_pos - current_pos
        rot_error = self.cul_rotation_error(target_rot, current_rot)
        # rot_error = np.zeros(3)
        # rot_error = -1*rot_error
        # print("pos error =", pos_error)
        # print("rot error =", rot_error) 
        error = np.concatenate([pos_error,rot_error])

        #达到目标点后不用重复计算，直接返回
        if (np.linalg.norm(pos_error) < self.position_tolerance and np.linalg.norm(rot_error)<self.rotation_tolerance):
            return robot.get_joint_pos()
        
        #ik
        dq_task = self.solve_task(J,error)
        # dq_null = self.solve_null_space(J,current_q,lower_limit,upper_limit)
        dq_null = self.solve_null_space_elbow(robot,J,current_q)
        dq = dq_task + dq_null

        dq = np.clip(dq,-max_dq,max_dq)

        q_target = current_q + dq*self.step_size
        q_target = np.clip(q_target,lower_limit,upper_limit)

        # self.cnt +=1
        # if self.cnt % 500 == 0:
        #     print("---------------------")
        #     print("pos_error =", np.linalg.norm(pos_error))
        # print("rot_error =", np.linalg.norm(rot_error))
        # print(np.linalg.norm(rot_error))
        #     print("cond =", np.linalg.cond(J))
        #     print("dq_task =", np.linalg.norm(dq_task))
        #     print("dq_null =", np.linalg.norm(dq_null))
        #     print("joint4 =", current_q[3])
        #     print("joint6 =", current_q[5])

        return q_target
    

    def solve_task(self,J,error):
        JT = J.T
        lambda2 = self.damping**2
        dq_task = JT @ np.linalg.solve(
            J @ JT + lambda2 * np.eye(6),
            error
        )

        return dq_task

    def solve_null_space(self,J,current_q,lower_limit,upper_limit):
        #null space
        J_pinv = np.linalg.pinv(J)
        N = np.eye(J.shape[1]) - J_pinv @ J 
        k_null = 0.02

        grad = self.joint_limit_gradient(current_q,lower_limit,upper_limit)
        z = -k_null*grad
        dq_null = N @ z

        return dq_null
    
    def solve_null_space_avoid(self,robot,J):
        J_pinv = np.linalg.pinv(J)
        N = np.eye(J.shape[1]) - J_pinv @ J 

        grad = self.manipulability_grad(robot)
        k = 0.2
        z = k*grad  #希望manipulability越大越好，所以找梯度
        dq_null = N @ z
        return dq_null

    def solve_null_space_elbow(self,robot,J,current_q):
        '''
        加入姿态约束ik后elbow老是伸直不弯曲所以给他一个弯曲的姿势
        '''
        #null space
        J_pinv = np.linalg.pinv(J)
        N = np.eye(J.shape[1]) - J_pinv @ J 
        k_null = 0.1

        q_pref = np.array([0.0,-0.785,0.0,-2.356,0.0,1.571,0.785])
        # q_pref = robot.home_q

        grad = current_q - q_pref
        z = -k_null*grad
        dq_null = N @ z

        return dq_null

    def cul_rotation_error(self,target_rot,current_rot):
        """
        输入:
            当前旋转矩阵 Rc
            目标旋转矩阵 Rd
        输出:
            旋转误差轴角
        """        
        R_error = (target_rot @ current_rot.T)
        rot = Rotation.from_matrix(R_error)
        return rot.as_rotvec()
    

    def compute_grasp_rotation(self, object_normal):
        """
        根据物体法向自动生成抓取姿态
        输入
        object_normal : (3,)
            例如桌面物体就是 [0,0,1]
        返回
        R_target : (3,3)
        """
        # ---------- Tool Z ----------
        tool_z = -object_normal.astype(float)
        tool_z /= np.linalg.norm(tool_z)

        # ---------- Reference Axis ----------
        ref = np.array([1.0, 0.0, 0.0])

        if abs(np.dot(ref, tool_z)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        # ---------- Tool Y ----------
        tool_y = np.cross(tool_z, ref)
        tool_y /= np.linalg.norm(tool_y)
        # ---------- Tool X ----------
        tool_x = np.cross(tool_y, tool_z)
        tool_x /= np.linalg.norm(tool_x)

        # ---------- Rotation ----------
        R_target = np.column_stack([
            tool_x,
            tool_y,
            tool_z
        ])

        return R_target
    
    def joint_limit_gradient(self,q,lower_limit,upper_limit):
        eps = 1e-3 #防止分母 = 0
        dist_min = np.maximum(q - lower_limit,eps)
        dist_max = np.maximum(upper_limit - q,eps)

        grad = (2.0/(dist_min ** 3) - 2.0/(dist_max ** 3))

        return grad

    def manipulability(self,robot,q):
        J = robot.compute_Jacoobian(q)
        s = np.linalg.svd(J, compute_uv=False)
        # print('w:',np.prod(s))
        return np.prod(s)
    
    def manipulability_grad(self,robot):
        eps = 1e-4
        current_q = robot.get_joint_pos()

        w0 = self.manipulability(robot,current_q)
        grad = np.zeros(len(current_q))
        for i in range(len(current_q)):
            q = current_q.copy()
            q[i] += eps
            w1 = self.manipulability(robot,q)
            grad[i] = (w1 - w0) /eps

        return grad

        



        