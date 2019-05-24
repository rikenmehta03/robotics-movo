import numpy as np
import cv2
from jaco import jaco_reacher

width = 640
height = 480

env = jaco_reacher.env


obs = env.reset()

# env.step(env.action_space.low)
# screen = env.render(mode='rgb_array')
# cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
# cv2.waitKey(0)

done = False
while not done:
  for i in range(5000):
    # action = env.action_space.sample()
    action = np.random.uniform(-20, 20, size=[9])
    new_obs, reward, done, info = env.step(action)
    screen = env.render(mode='rgb_array')
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if(cv2.waitKey(25) & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        break
