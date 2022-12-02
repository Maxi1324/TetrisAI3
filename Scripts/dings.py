
# %%
trainstep = 1000000
learningrate = 1e-3
warmup = 100

rewardHoch = 4
rewardmult = 2
alwaysReward = 0.000000001

render = True

verbose = 3000

speed = 3

loadfile = None
savefile = "BoltzTiny"

# %%
import numpy as np 

import gym
import random


from gym import Env, spaces
import time
import pygame
from pygame import gfxdraw

import os

class TetrisBlock:
    blockColors = [
        (255, 0, 0), # 1 = red
        (0, 255, 0), # 2 = green
        (0, 0, 255), # 3 = blue
        (255, 255, 0), # 4 = yellow
        (255, 0, 255), # 5 = magenta
        (0, 255, 255), # 6 = cyan
        (255, 255, 255), # 7 = white
        (255, 127, 0), # 8 = orange
    ]
    blocktypes = [
      [[0,1,0,0],
       [0,1,0,0],
       [0,1,0,0],
       [0,1,0,0]],

      [[0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0]],

      [[0,0,0,0],
        [0,1,0,0],
        [0,1,1,0],
        [0,0,1,0]],

      [[0,0,0,0],
        [0,0,1,0],
        [0,1,1,0],
        [0,1,0,0]],

      [[0,0,0,0],
        [0,1,1,0],
        [0,0,1,0],
        [0,0,1,0]],

      [[0,0,0,0],
        [0,0,1,0],
        [0,1,1,0],
        [0,0,1,0]],

      [[0,0,0,0],
        [0,1,1,0],
        [0,1,0,0],
        [0,1,0,0]]
    ]

    rotateN = 0

    offset = 0

    def __init__(self, blockType, x, y):
        self.x = x
        self.y = y
        self.blocktype = blockType

        self.block = self.blocktypes[blockType]

        self.static = False
        self.offsets =  np.zeros(4)

        self.blockColor = self.blockColors[blockType]
    
    def absolutPositions(self):
      block = self.block
      if(not self.static):
        block = self.calcRotBlock(self.rotateN, self.block)
      return self.calcAbsolutPos(block)
    
    def offset(self,y):
        a = self.calcAbsolutPos(self.block)
        for el, i in enumerate(a):
            elx,ely = el
            if ely == y:
                self.offsets[i] += 1

    def calcAbsolutPos(self, block):
        positions = []
        for i in range(4):
            for j in range(4):
              if block[i][j] == 1:
                  positions.append((self.x+i, self.y+j+self.offsets[j]))
        return positions

    def calcRotBlock(self, rot,block):
      rotatedBlock = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
      for i in range(0,4):
        for j in range(0, 4):
              x = i
              y = j
              if rot == 1:
                x = j
                y = 3-i
              elif rot == 2:
                x = 3-i
                y = 3-j
              elif rot == 3:
                x = 3-j
                y = i
              rotatedBlock[i][j] = block[x][y]
      return rotatedBlock
              
    def clearLocalLine(self,y):
        for i in range(4):
            for j in range(4):
                if self.y+j+self.offsets[j] == y:
                    self.block[i][j] = 0

    def getColor(self):
        return self.blockColor

    def rotate(self):
        self.rotateN = self.calcNextRot()
    
    def calcNextRot(self):
      return (self.rotateN + 1) % 4

    def makeStatic(self):
      self.static = True
      self.block = self.calcRotBlock(self.rotateN, self.block)

class TetrisE(Env):

  def __init__(self) -> None:
    super(TetrisE,self).__init__()
    #left, right, down,nothing, rotate
    self.action_space = spaces.Discrete(5)
    self.observation_space = spaces.Box(low=0, high=1,shape=(20,10,1),dtype=np.uint8)
    
    self.doRender = render

    self.reset()

  def step(self,action):
    GameField = self.gameField()
    if action == 0:
        self.move(-1,0,GameField)
    elif action == 1:
        self.move(1,0,GameField)
    elif action == 2:
        self.move(0,1,GameField)
    elif action == 3:
        self.rotate(GameField)

    self.count += 1

    if(self.count % speed == 0):
      self.move(0,1,GameField)

    lines = self.clearLines(GameField)

    if(self.doRender):
      self.render()

    return self.GameFieldToObservation(GameField), self.calcReward(lines), self.lost(GameField), {}

  def calcReward(self, linewCleared):
    a = pow(linewCleared*rewardmult,rewardHoch )
    return a +alwaysReward

  def GameFieldToObservation(self, gameField):
    observation = np.zeros((20,10,1))
    for i in gameField:
      if(i[1] >= 0 and i[1] < 20 and i[0]>= 0 and i[0]<10):
        observation[i[1]][i[0]] = 1
    return observation

  def reset(self):
    self.count = 0
    self.blocksCount = 0
    self.blocks = []
    self.newBlock()

    if self.doRender:
      pygame.quit()
      pygame.display.init()
      self.screen = pygame.display.set_mode((300, 600))
      self.screen.fill((0, 0, 0))
      pygame.display.flip()

    return self.GameFieldToObservation(self.gameField())

  def render(self, mode='human', close=False):
    self.screen.fill((0, 0, 0))
    for i in self.blocks:
      for j in i.absolutPositions():
        x,y = j
        pygame.draw.rect(self.screen, i.getColor(), (x*30, y*30, 30, 30))

    pygame.display.flip()

  def newBlock(self):
      random.seed(self.blocksCount)
      self.blocksCount+=1
      self.CurrentBlock = TetrisBlock(random.randint(0,6), 4, -1)
      self.blocks.append(self.CurrentBlock)

  def lost(self, gameField):
    for i in gameField:
      if i[1] == 0:
        return True 
    return False

  def stop(self):
    pygame.quit()
  
  def clearLines(self,gameField):
    lines = []
    for j in range(0,20):
      cleared = True
      for i in range(0,10):
        if (i,j) not in gameField:
          cleared = False
      if cleared:
        lines.append(j)

    for i in lines:
      for j in self.blocks:
        j.clearLocalLine(i)

    for line in lines:
        for i in range(line, 0, -1):
            for j in self.blocks:
                j.offset(i)

    return len(lines)

  def move(self,x,y,gameField):
    currentBlock = self.CurrentBlock
    absolutePositions = currentBlock.absolutPositions()

    for i in absolutePositions:
      newPos = (i[0]+x,i[1]+y)
      if (i[0],i[1]+1) in gameField:
        self.CurrentBlock.makeStatic()
        self.newBlock()
        return
      elif i[1]+y > 19 or i[1]+y < -1:
        self.CurrentBlock.makeStatic()
        self.newBlock()
        return
      elif i[0]+x > 9 or i[0]+x < 0 or newPos in gameField:
        return
    self.CurrentBlock.x += x
    self.CurrentBlock.y += y
  
  def allowRotate(self,gameField):
    currentBlock = self.CurrentBlock
    absolutePositions = currentBlock.calcAbsolutPos(currentBlock.calcRotBlock(currentBlock.calcNextRot(),currentBlock.blocktypes[currentBlock.blocktype]))
    for i in absolutePositions:
      if i[0] > 9 or i[0] < 0 or i in gameField or i[1] > 19:
        return False
    return True	

  def rotate(self,gameField):
    if(self.allowRotate(gameField)):
      self.CurrentBlock.rotate()

  def gameField(self):
    gf=[]
    for i in self.blocks:
      if(self.CurrentBlock != i):
        for j in i.absolutPositions():
          gf.append(j)
    return gf

# %%
from datetime import datetime
import time
data = []

for i in range(0,10):
    env = TetrisE()
    observation, reward, done= env.reset(), 0, False
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
        time.sleep(.15) 
        data.append([observation,action])
        observation, reward, done, info = env.step(action)
        
datat =  datetime.now().strftime("%m%d%Y_%H%M%S")
np.save("data "+datat+".npy",np.array(data, dtype=np.int32))
        



