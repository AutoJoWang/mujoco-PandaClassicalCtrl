import mujoco
import mujoco.viewer
import os
import numpy as np


class MujocoEnv:

    def __init__(self, xml_path):

        self.model = mujoco.MjModel.from_xml_path(
            xml_path
        )

        self.data = mujoco.MjData(
            self.model
        )

        self.home_q = np.array([0.0,-0.785,0.0,-2.356,0.0,1.571,0.785])
        self.data.qpos[:7] = self.home_q
        self.data.ctrl[:7] = self.home_q
        self.data.ctrl[7] = 255
        mujoco.mj_forward(self.model,self.data)

        self.site_ids={}
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,i)
            self.site_ids[name] = i
        
        self.planner_dt = 0.5  
        self.control_dt = 0.01  #100hz
        self.physics_dt = 0.002 #500hz

        self.camera_name = 'top_camera'
        self.render = mujoco.Renderer(self.model,height=240,width=320)




    def step(self):

        mujoco.mj_step(
            self.model,
            self.data
        )

    def launch_viewer(self):

        return mujoco.viewer.launch_passive(
            self.model,
            self.data
        )

    def reset(self):

        mujoco.mj_resetData(
            self.model,
            self.data
        )
        self.data.qpos[:7] = self.home_q
        self.data.ctrl[:7] = self.home_q
        self.data.ctrl[7] = 255

        mujoco.mj_forward(
            self.model,
            self.data
        )

    def render_camera(self):

        self.render.update_scene(
            self.data,
            camera=self.camera_name
        )

        rgb = self.render.render()

        return rgb
    
    def save_camera_image(self,path="camera.png"):

        import imageio

        img = self.render_camera()

        imageio.imwrite(
            path,
            img
        )

    def set_site_position(self,name,pos):
        site_id = self.site_ids[name]
        self.model.site_pos[site_id] = pos