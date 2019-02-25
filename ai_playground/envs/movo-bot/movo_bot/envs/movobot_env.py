import os
import math
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

from pkg_resources import parse_version

from . import movo


class MovobotEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        pass

    def init(self, render=True, discrete_reward=True):
        self._timeStep = 1./240.
        self._observation = []
        self._renders = render
        self.discrete_reward = discrete_reward
        self._envStepCounter = 0
        self._max_episode_steps = 128
        self._p = p

        if (render):
            self.physicsClient = self._p.connect(p.GUI)
        else:
            self.physicsClient = self._p.connect(
                p.DIRECT)  # non-graphical version

        self._p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF

        self._view_mat, self._proj_mat = self._get_camera_matrices()

        self._width = 341
        self._height = 256

        action_dim = 3
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-1 * action_high, action_high)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self._height*self._width*3,))
        self._seed()
        self._p.setTimeStep(self._timeStep)

        self._p.resetSimulation()
        self._p.setPhysicsEngineParameter(numSolverIterations=150)
        self._p.setTimeStep(self._timeStep)
        self._p.loadURDF("plane.urdf")
        self._p.setGravity(0, 0, -10)
        self.tableId = self._p.loadURDF(
            "table/table.urdf", [1.000000, 0.00000, 0.000000], [0.000000, 0.000000, 0.0, 1.0])

        self._movo = None
        self.blockUid = None

    def _reset(self):
        self._envStepCounter = 0
        self.terminated = 0

        xpos = 0.5 + 0.2*random.random()
        ypos = 0 + 0.25*random.random()
        ang = 3.1415925438*random.random()
        orn = self._p.getQuaternionFromEuler([0, 0, ang])
        pos = [xpos, ypos, 0.63]
        if self.blockUid is None:
            self.blockUid = self._p.loadURDF(
                "block.urdf", pos, orn)
        else:
            self._p.resetBasePositionAndOrientation(self.blockUid, pos, orn)

        if self._movo is None:
            self._movo = movo.Movo()
        else:
            self._movo.reset()
        self._p.stepSimulation()
        self._get_observation()
        return self._observation

    def _get_camera_matrices(self):
        eyePos = (0.668, -0.033249524857753725, 1.061325139221183)
        targetPos = (1.116, 0.0, 0.0)
        upVector = (0.0, 0.0, 1.0)
        viewMat = self._p.computeViewMatrix(eyePos, targetPos, upVector)
        projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -
                      1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
        return viewMat, projMatrix

    def __del__(self):
        self._p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action=None):
        self._movo.step([action[0], action[1], action[2]])
        self._p.stepSimulation()
        self._envStepCounter += 1
        reward = self._reward()
        done = self._termination()
        self._get_observation()
        return self._observation, reward, done, {}

    def _render(self, mode='human', close=False):
        pass

    def _reward(self):
        if not self.discrete_reward:
            table_contact_points = self._p.getContactPoints(
                self.tableId, self._movo.movoId)
            # if hit the table 
            if (len(table_contact_points) > 0):
                return -1000
            block_pos = np.array(
                self._p.getBasePositionAndOrientation(self.blockUid)[0])
            reach_position = np.array(self._p.getLinkState(
                self._movo.movoId, self._movo.movoEndEffectorIndex)[0])

            reward = -np.linalg.norm(block_pos - reach_position)
        else:
            tableContactPoints = self._p.getContactPoints(
                self.tableId, self._movo.movoId)
            blockContactPoints = self._p.getContactPoints(
                self.blockUid, self._movo.movoId)

            if (len(tableContactPoints)):
                reward = -1.0
            elif (len(blockContactPoints)):
                reward = 1.0
            else:
                reward = 0.0

        return reward

    def _termination(self):
        if (self.terminated or self._envStepCounter > self._max_episode_steps):
            self._get_observation()
            return True

        tableContactPoints = self._p.getContactPoints(
            self.tableId, self._movo.movoId)
        blockContactPoints = self._p.getContactPoints(
            self.blockUid, self._movo.movoId)
        if (len(tableContactPoints) or len(blockContactPoints)):
            self.terminated = 1
            return True
        return False

    def _get_observation(self):
        image_blob = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_mat,
            projectionMatrix=self._proj_mat)[2]
        image_rgb = np.reshape(image_blob, (self._height, self._width, 4))[:, :, :3]
        self._observation = image_rgb.astype(np.uint8)