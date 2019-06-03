# import torch
import math

import random
import sys
import time

from kinova_api import kinova
from jaco import jaco_reacher

import numpy as np

import math
import cv2


env = jaco_reacher.env


def render():
    screen = env.render(mode='rgb_array')
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    cv2.waitKey(10)

def get_jaco_angles():
    pos = kinova.get_angular_position()
    angles = [a for a in pos.Actuators] + [a for a in pos.Fingers]
    return angles


gears = env.dmcenv.physics.model.actuator_gear[:, 0]

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

def move_mujoco_to_real(angles):
    env.dmcenv.physics.named.data.qpos[:9] = real_to_sim(angles)

kinova.start()


def get_observation_real():
    """Returns an observation of the (bounded) physics state."""
    obs = {}
    # obs['position'] = kinova.get_angular_position()
    obs['position'] = get_jaco_angles()
    obs['to_target'] = [0]
    return obs

prev_angles = get_jaco_angles()
move_mujoco_to_real(prev_angles)

while True:
    for i in range(100):
        angles = get_jaco_angles()

        if not np.allclose(prev_angles, angles):
            move_mujoco_to_real(angles)
            obs, reward, done, info = env.step(real_to_sim(angles))
            print(info)
        
        prev_angles[:] = angles
        # sometimes mujoco flips out and resets the sim's angles to all zeros
        if np.allclose(np.zeros(9), env.dmcenv.physics.named.data.qpos[:9]):
            env.step(sim_to_real(angles))

        render()

cv2.destroyAllWindows()
