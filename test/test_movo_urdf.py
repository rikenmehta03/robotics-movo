import pybullet as p
import time
import pybullet_data
import cv2
import numpy as np

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
sphereId = p.createVisualShape(shapeType=p.GEOM_SPHERE,radius=0.05,fileName="sphere_smooth.obj", rgbaColor=[1,0,0,1])
p.createMultiBody(baseVisualShapeIndex = sphereId, basePosition = [1.0, 0.0, 0.0])
movoId = p.loadURDF("../movo_urdf/movo.urdf", useFixedBase=True)



# num_joints = p.getNumJoints(movoId)
# for jId in range(num_joints):
#     joint = p.getJointInfo(movoId, jId)
#     print(joint[1])
#     link = p.getLinkState(movoId,jId)
#     print(link[0],p.getEulerFromQuaternion(link[1]))

eyePos = (0.668, -0.033249524857753725, 1.061325139221183)
targetPos = (1.116, 0.0, 0.0)
upVector = (0.0, 0.0, 1.0)
viewMat = p.computeViewMatrix(eyePos, targetPos, upVector)
projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]

# tPosX = p.addUserDebugParameter("targetPosX",0,2,0)
# ePosX = p.addUserDebugParameter("eyePosX", 0, 1, 0.3095777753273074)

while True:    
    # viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
    # projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    # eyePos = (p.readUserDebugParameter(ePosX), -0.033249524857753725, 1.061325139221183)
    # targetPos = (p.readUserDebugParameter(tPosX), 0.0, 0.0)   
    img_arr = p.getCameraImage(width=341,height=256, viewMatrix=viewMat, projectionMatrix=projMatrix)
    # img = img_arr[2]
    # img = np.reshape(img, (341, 256, 4))
    p.stepSimulation()
    time.sleep(1./240.)