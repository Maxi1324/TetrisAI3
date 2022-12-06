# %%

from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

# %%
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from rl.policy import BoltzmannQPolicy

trainstep = 1000000
learningrate = 1e-3
warmup = 100

rewardHoch = 4
rewardmult = 2
alwaysReward = 0.000000001

render = False

verbose = 3000

speed = 10

loadfile = None
savefile = "BoltzTiny"

# %%



print(model.summary())
    
policy = BoltzmannQPolicy() 

# %%
import numpy as np 

import gym
import random

from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from gym import Env, spaces
import time
import pygame
from pygame import gfxdraw

import os



# %%






# %%
from datetime import datetime

tet = TetrisE()

memory = SequentialMemory(limit=50000, window_length=1)


dqn = DQNAgent(model=model, nb_actions=5, memory=memory, nb_steps_warmup=warmup,
               target_model_update=1e-2, policy=policy)

if(loadfile):
    dqn.load_weights(loadfile)

dqn.compile(Adam(learning_rate=learningrate))

dqn.fit(tet, nb_steps=trainstep, visualize=False, verbose=1000)

if(savefile):
    datat =  datetime.now().strftime("%m%d%Y_%H%M%S")

    folder = savefile+"_"+datat+"/"
    os.mkdir(folder)
    dqn.save_weights(folder+savefile+".h5f", overwrite=True)

tet.stop()

print("finished training")




