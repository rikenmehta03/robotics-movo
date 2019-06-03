# import torch
import math

import random
import sys
import time

from kinova_api import kinova

import numpy as np

from dm_control import suite
from jaco import jaco
import pyglet
import math

import inspect
import cv2
from dm_control.mujoco.wrapper.mjbindings import mjlib


LOCAL_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}
suite._DOMAINS.update(LOCAL_DOMAINS)
env = suite.load(domain_name="jaco", task_name="basic")

action_spec = env.action_spec()
time_step = env.reset()

action = np.zeros([9])
time_step = env.step(action)

def render():
    width = 640
    height = 480
    screen = env.physics.render(height, width, camera_id=0)    
    cv2.imshow('window', screen)
    cv2.waitKey(10)

def get_jaco_angles():
    pos = kinova.get_angular_position()
    angles = [a for a in pos.Actuators] + [a for a in pos.Fingers]
    return angles

def move_mujoco_to_real():
    with env.physics.reset_context():
        angles = get_jaco_angles()
        env.physics.named.data.qpos[:9] = real_to_sim(angles)

gears = env.physics.model.actuator_gear[:, 0]

# where the robot has to be (in kinova coordinates)
# to be at zero in mujoco
zero_offset = np.array([-180, 270, 90, 180, 180, -90, 0, 0, 0])

# correct for the different physical directions of a +theta
# movement between mujoco
directions = np.array([-1, 1, -1, -1, -1, -1, 1, 1, 1])

# correct for the degrees -> radians shift going from arm
# to mujoco
scales = np.array([math.pi / 180] * 6 + [0.78 / 6800] * 3)

def real_to_sim(angles):
    return (angles - zero_offset) * directions * scales
  
def sim_to_real(angles):
      return (angles / (directions * scales)) + zero_offset

kinova.start()

# controls = np.array(env.physics.named.data.qpos[:6])
print('starting position: ')
print(env.physics.named.data.qpos[:9])
# controls[0] = 0

def get_observation_real():
    """Returns an observation of the (bounded) physics state."""
    obs = {}
    #obs['position'] = kinova.get_angular_position()
    obs['position'] = get_jaco_angles()
    obs['to_target'] = [0]
    return obs

move_mujoco_to_real()
while True:
    for i in range(100):
        angles = get_jaco_angles()
        angles_real = get_observation_real()
        print(env.step(real_to_sim(angles)))
        print(env.physics.named.data.geom_xpos['target'])
        move_mujoco_to_real()

        # sometimes mujoco flips out and resets the sim's angles to all zeros
        if np.allclose(np.zeros(9), env.physics.named.data.qpos[:9]):
            move_mujoco_to_real()

        render()

cv2.destroyAllWindows()
