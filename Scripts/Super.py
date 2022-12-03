import Env
import numpy as np
import pygame
from datetime import datetime
import time

data = []
label = []

for i in range(0, 10):
    env = Env.TetrisE()
    retObs, reward, done = env.reset(True), 0, False
    while not done:
        action = 4
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.stop()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                if event.key == pygame.K_RIGHT:
                    action = 1
                if event.key == pygame.K_DOWN:
                    action = 2
                if event.key == pygame.K_UP:
                    action = 3
        time.sleep(.1)
        data.append(retObs)
        label.append(np.array(action))
        retObs, reward, done, info =  env.step(action)
        env.render()
        if(done):
            env.reset()

datat = datetime.now().strftime("%m%d%Y_%H%M%S")
np.save("data "+datat+".npy", np.array(data)) 
np.save("label "+datat+".npy", np.array(label)) 