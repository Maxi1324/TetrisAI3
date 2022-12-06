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


from datetime import datetime

from Env.Env import TetrisE
from models.DenseTiny import model

from rl.policy import BoltzmannQPolicy


tet = TetrisE()

memory = SequentialMemory(limit=50000, window_length=1)

policy = BoltzmannQPolicy() 

dqn = DQNAgent(model=model, nb_actions=5, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)


if(loadfile):
    dqn.load_weights(loadfile)

dqn.compile(Adam(learning_rate=learningrate))

dqn.fit(tet, nb_steps=trainstep, visualize=False, verbose=1000)