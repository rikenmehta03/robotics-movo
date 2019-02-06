import pybullet as p
import time
import pybullet_data
import cv2
import numpy as np
import json

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
# sphereId = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.05,fileName="sphere_smooth.obj", rgbaColor=[1,0,0,1])
# p.createMultiBody(baseVisualShapeIndex = sphereId, basePosition = [1.0, 0.0, 0.0])
movoId = p.loadURDF("../movo_urdf/movo.urdf", useFixedBase=True)



num_joints = p.getNumJoints(movoId)
joint_dict = {}
for jId in range(num_joints):
    joint = p.getJointInfo(movoId, jId)
    # print(joint)

    # joint_dict[joint[1].decode('ascii')] = {
    #     'jointIndex': joint[0],
    #     'jointType': joint[2],
    #     'qIndex': joint[3],
    #     'uIndex': joint[4]
    # }

    joint_dict[joint[0]] = {
        'jointName': joint[1].decode('ascii'),
        'jointType': joint[2],
        'qIndex': joint[3],
        'uIndex': joint[4]
    }
    # link = p.getLinkState(movoId,jId)
    # print(link[0],p.getEulerFromQuaternion(link[1]))

# with open('movo_joints_dict.json', 'w') as fp:
#     json.dump(joint_dict, fp, indent=4)

# eyePos = (0.668, -0.033249524857753725, 1.061325139221183)
# targetPos = (1.116, 0.0, 0.0)
# upVector = (0.0, 0.0, 1.0)
# viewMat = p.computeViewMatrix(eyePos, targetPos, upVector)
# projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

# tPosX = p.addUserDebugParameter("targetPosX",0,2,0)
# ePosX = p.addUserDebugParameter("eyePosX", 0, 1, 0.3095777753273074)

xId = p.addUserDebugParameter("X", 0, 3, 1.0)
yId = p.addUserDebugParameter("Y", -3, 3, 0.0)
zId = p.addUserDebugParameter("Z", -3, 3, 1.0)


# print(leftGripperPosition)
while True:    
    # viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
    # projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    # eyePos = (p.readUserDebugParameter(ePosX), -0.033249524857753725, 1.061325139221183)
    # targetPos = (p.readUserDebugParameter(tPosX), 0.0, 0.0)   
    # img_arr = p.getCameraImage(width=341,height=256, viewMatrix=viewMat, projectionMatrix=projMatrix)
    # img = img_arr[2]
    # img = np.reshape(img, (341, 256, 4))
    # input('press any key to continue')
    # leftGripperPosition -= 0.1
    # p.setJointMotorControl2(movoId, 37, controlMode=p.POSITION_CONTROL, targetPosition=leftGripperPosition, force = 200.)
    targetPosition = (p.readUserDebugParameter(xId), p.readUserDebugParameter(yId), p.readUserDebugParameter(zId))
    jointPositions = p.calculateInverseKinematics(movoId, 20, targetPosition)
    c = 0
    for i in range(num_joints):
        if joint_dict[i]['jointType'] != 4:
            print(joint_dict[i]['jointName'])
            print(jointPositions[c])
            p.setJointMotorControl2(movoId, i, controlMode=p.POSITION_CONTROL, targetPosition=jointPositions[c], targetVelocity=0)
            c+=1
    p.stepSimulation()
    time.sleep(1./240.)
    