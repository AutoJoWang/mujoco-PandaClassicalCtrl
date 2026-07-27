import mujoco
import numpy as np

from simulation.mujoco_env import MujocoEnv
from robot.panda import PandaRobot

XML_PATH = "robot_description/franka_emika_panda/scene.xml"

env = MujocoEnv(XML_PATH)
robot = PandaRobot(env)
mujoco.mj_forward(env.model, env.data)
# 获取末端body id
body_id = robot.ee_body_id

# Jacobian
jacp = np.zeros((3, env.model.nv))
jacr = np.zeros((3, env.model.nv))

mujoco.mj_jacBody(
    env.model,
    env.data,
    jacp,
    jacr,
    body_id
)

print("Position Jacobian")
print(jacp)

print()

print("Rotation Jacobian")
print(jacr)