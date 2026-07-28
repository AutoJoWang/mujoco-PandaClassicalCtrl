import mujoco
import numpy as np


class Camera:

    def __init__(
        self,
        env,
        camera_name="top_camera"
    ):

        self.model = env.model
        self.data = env.data
        self.buffer = []

        self.camera_name = camera_name

        self.cam_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name
        )

        if self.cam_id < 0:
            raise ValueError(
                f"Camera {camera_name} not found"
            )


        self.width = 320
        self.height = 240


        self.renderer = mujoco.Renderer(
            self.model,
            height=self.height,
            width=self.width
        )


        # camera intrinsic
        self.fovy = self.model.cam_fovy[self.cam_id]


        self.fx = None
        self.fy = None
        self.cx = self.width/2
        self.cy = self.height/2


        self.compute_intrinsic()



    def compute_intrinsic(self):

        """
        MuJoCo fovy转换焦距
        """

        fovy_rad = np.deg2rad(
            self.fovy
        )

        self.fy = (
            self.height/2
        ) / np.tan(
            fovy_rad/2
        )

        self.fx = self.fy



    def get_rgb(self):

        """
        返回 RGB
        """

        self.renderer.update_scene(
            self.data,
            camera=self.camera_name
        )


        rgb = self.renderer.render()

        return rgb.copy()



    def get_depth(self):

        """
        返回depth图
        """

        self.renderer.enable_depth_rendering()


        self.renderer.update_scene(
            self.data,
            camera=self.camera_name
        )

        depth = self.renderer.render()

        self.renderer.disable_depth_rendering()

        return depth.copy()



    def pixel_to_camera(
            self,
            u,
            v,
            depth
    ):

        a = (u-self.cx)/self.fx
        b = (v-self.cy)/self.fy

        z = depth / np.sqrt(
            a*a+b*b+1
        )

        x = a*z
        y = b*z


        # MuJoCo camera frame
        return np.array([
            x,
            y,
            -z
        ])


    # def pixel_to_camera(
    #     self,
    #     u,
    #     v,
    #     depth
    # ):

    #     x = (u-self.cx)/self.fx * depth
    #     y = -(v-self.cy)/self.fy * depth
    #     z = depth


    #     return np.array([
    #         x,
    #         y,
    #         -z
    #     ])

    def camera_to_world(
        self,
        point_camera
    ):

        """
        camera frame
        ->
        world frame
        """
        #旋转矩阵
        R = self.data.cam_xmat[
            self.cam_id
        ].reshape(3,3)


        t = self.data.cam_xpos[
            self.cam_id
        ]


        point_world = (
            R @ point_camera
            +
            t
        )

        return point_world
    
    def get_object_world_pos(self,pixel):
        u,v = pixel
        depth_img = self.get_depth()

        depth = depth_img[v,u]

        point_camera = self.pixel_to_camera(u,v,depth)
        point_world = self.camera_to_world(point_camera)
        # print('pixel',pixel,'depth',depth)

        return point_world
    
    def depth_to_meters(self,depth):

        near = self.model.vis.map.znear
        far = self.model.vis.map.zfar

        return (near*far / (far -depth*(far-near)))
    
    def filter(self,pos):

        self.buffer.append(pos)

        if len(self.buffer)>10:
            self.buffer.pop(0)

        return np.mean(
            self.buffer,
            axis=0
        )
    
    def debug_pose(self):

        R = self.data.cam_xmat[
            self.cam_id
        ].reshape(3,3)

        t = self.data.cam_xpos[
            self.cam_id
        ]

        print("================")
        print("camera pos:")
        print(t)

        print("camera rot:")
        print(R)

        print("camera forward:")
        print(R[:,2])

        print("================")

    def debug_world_to_camera(
            self,
            cube_world
    ):

        R = self.data.cam_xmat[
            self.cam_id
        ].reshape(3,3)


        t = self.data.cam_xpos[
            self.cam_id
        ]


        cube_cam = R.T @ (
            cube_world - t
        )

        print(
            "cube in camera frame:",
            cube_cam
        )

        return cube_cam
    
    def world_to_pixel(self, point):

        R=self.data.cam_xmat[self.cam_id].reshape(3,3)
        t=self.data.cam_xpos[self.cam_id]

        pc=R.T@(point-t)

        x,y,z=pc

        u=self.fx*x/(-z)+self.cx
        v=self.fy*y/(-z)+self.cy

        return np.array([u,v])