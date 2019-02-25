import os
from collections import OrderedDict, namedtuple
import pybullet as p
import numpy as np

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

class Movo():
    def __init__(self, arm='right'):
        self.movoEndEffectorIndex = 23
        self.movoId = p.loadURDF(os.path.join(os.path.dirname(
            __file__), 'movo_urdf', 'movo.urdf'), useFixedBase=True)

        self.numJoints = p.getNumJoints(self.movoId)

        self.jointsInfo = OrderedDict()
        for jId in range(self.numJoints):
            joint = p.getJointInfo(self.movoId, jId)
            self.jointsInfo[joint[1].decode('ascii')] = JointInfo(*joint)

        p.resetBasePositionAndOrientation(
            self.movoId, [-0.100000, 0.000000, 0.000000], [0.000000, 0.000000, 0.000000, 1.000000])
        
        p.resetJointState(
            self.movoId, self.jointsInfo['linear_joint'].jointIndex, self.jointsInfo['linear_joint'].jointUpperLimit)

        self.resetState = p.getJointStates(self.movoId, range(self.numJoints))
    

    def _join(self, params):
        return '_'.join(params)

    def _set_joint_max_position(self, joint_name):
        p.setJointMotorControl2(self.movoId, self.jointsInfo[joint_name].jointIndex, controlMode=p.POSITION_CONTROL,
                                targetPosition=self.jointsInfo[joint_name].jointUpperLimit, targetVelocity=0)

    def _set_joint_position(self, joint_name, position):
        p.setJointMotorControl2(self.movoId, self.jointsInfo[joint_name].jointIndex, controlMode=p.POSITION_CONTROL,
                                targetPosition=position, targetVelocity=0)

    def _set_end_effector(self, endEffectorPosition):
        jointPositions = p.calculateInverseKinematics(
            self.movoId, self.movoEndEffectorIndex, endEffectorPosition)
        jointsInfo = list(self.jointsInfo.values())
        c = 0
        for i in range(self.numJoints):
            if jointsInfo[i].jointType != 4:
                p.setJointMotorControl2(
                    self.movoId, i, controlMode=p.POSITION_CONTROL, targetPosition=jointPositions[c], targetVelocity=0)
                c += 1

    def _reset_end_effector(self, endEffectorPosition):
        jointPositions = p.calculateInverseKinematics(
            self.movoId, self.movoEndEffectorIndex, endEffectorPosition)
        jointsInfo = list(self.jointsInfo.values())
        c = 0
        for i in range(self.numJoints):
            if jointsInfo[i].jointType != 4:
                p.resetJointState(
                    self.movoId, i, jointPositions[c])
                c += 1

    def reset(self):
        for i, _s in enumerate(self.resetState):
            p.resetJointState(self.movoId, i, _s[0])

    

    def step(self, action):
        low = np.array([0.3, -0.1, 0.63])
        high = np.array([0.7, 0.3, 0.8])
        if action is not None:
            pos = np.array(p.getLinkState(
                self.movoId, self.movoEndEffectorIndex)[0])
            pos = np.clip(pos + action, low, high)
            self._set_end_effector(pos)
