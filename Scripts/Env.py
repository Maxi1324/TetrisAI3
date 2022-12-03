
# %%
import numpy as np
from datetime import datetime
import os

import time
from gym import Env, spaces
import random
import gym
trainstep = 1000000
learningrate = 1e-3
warmup = 100

rewardHoch = 4
rewardmult = 2
alwaysReward = 0.000000001

render = False

verbose = 3000

speed = 3

loadfile = None
savefile = "BoltzTiny"

# %%

if(render):
    from pygame import gfxdraw
    import pygame

class TetrisBlock:
    blockColors = [
        (255, 0, 0),  # 1 = red
        (0, 255, 0),  # 2 = green
        (0, 0, 255),  # 3 = blue
        (255, 255, 0),  # 4 = yellow
        (255, 0, 255),  # 5 = magenta
        (0, 255, 255),  # 6 = cyan
        (255, 255, 255),  # 7 = white
        (255, 127, 0),  # 8 = orange
    ]
    blocktypes = [
        [[0, 1, 0, 0],
         [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]],

        [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]],

        [[0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0]],

        [[0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0]],

        [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]],

        [[0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0]],

        [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]]
    ]

    rotateN = 0

    def __init__(self, blockType, x, y):
        self.x = x
        self.y = y
        self.blocktype = blockType
        self.blockColor = self.blockColors[blockType]

    def absolutPositions(self):
        block = self.calcRotBlock(self.rotateN, self.blocktypes[self.blocktype])
        return self.calcAbsolutPos(block)

    def calcAbsolutPos(self, block):
        positions = []
        for i in range(4):
            for j in range(4):
                if block[i][j] == 1:
                    positions.append((self.x+i, self.y+j))
        return positions

    def calcRotBlock(self, rot, block):
        rotatedBlock = np.zeros((4, 4), dtype=np.uint8)
        for i in range(0, 4):
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

    def getColor(self):
        return self.blockColor

    def rotate(self):
        self.rotateN = self.calcNextRot()

    def calcNextRot(self):
        return (self.rotateN + 1) % 4


class TetrisE(Env):

    def __init__(self) -> None:
        super(TetrisE, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 10, 1), dtype=np.uint8)
        self.reward_range = (-1, 1)
        self.doRender = render

    def step(self, action):
        self.linedCleared = 0
        self.count += 1
        observation = self.getObservation()

        if action == 0:
            self.moveBlock(-1,0)
        elif action == 1:
            self.moveBlock(1,0)
        elif action == 2:
            runter = self.moveBlock(0,1)
            while(runter):
                runter = self.moveBlock(0,1)
        elif action == 3:
            self.rotate(observation)

        self.moveBlock(0,1)

        retObs = observation.reshape(20,10,1)
        reward = self.calcReward(self.linedCleared)
        done = self.lost()
        info = {}

        if(self.doRender):
            self.render()

        return retObs, reward, done, info
        
    def calcReward(self, linewCleared):
        a = pow(linewCleared*rewardmult,rewardHoch )
        return a +alwaysReward

    def clearLines(self):	
        lines = []
        for j in range(0,20):
            cleared = True
            for i in range(0,10):
                if not self.inField(i,j):
                    cleared = False
            if cleared:
                lines.append(j)

        remove = []
        for line in lines:
            for i,bl in enumerate(self.GameField):
                x,y,col = bl
                if y == line:
                    remove.append(bl)
        
        for r in remove:
            self.GameField.remove(r)    

        for line in lines:
            for i,bl in enumerate(self.GameField):
                x,y,col = bl
                if y < line:
                    self.GameField[i] = (x,y+1,col)
        
        self.linedCleared = len(lines)
             

    def lost(self):
        for pos in self.GameField:
            x,y,col = pos
            if y == 1:
                return True
        return False

    def reset(self):
        self.count = 0
        self.BlockCount = 0
        self.GameField = []
        self.newBlockWithoutSave()


        if self.doRender:
            pygame.quit()
            pygame.display.init()
            self.screen = pygame.display.set_mode((300, 600))
            self.screen.fill((0, 0, 0))
            pygame.display.flip()
        return np.zeros((20,10,1), dtype=np.uint8)

    def moveBlock(self, x, y):
        for pos in self.CurrentBlock.absolutPositions():
            posX, posY = pos
            newPos = (posX+x,posY+y)
            if (self.inField(posX,posY+1)) or ( posY+y > 19 or posY+y < -1):
                self.newBlock()
                self.clearLines()
                return False                
            elif posX+x > 9 or posX+x < 0 or self.inField(posX+x,posY+y):
                return False    

        self.CurrentBlock.x += x
        self.CurrentBlock.y += y
        return True
    
    def inField(self, x, y):
        for pos in self.GameField:
            posX, posY,col = pos
            if posX == x and posY == y:
                return True
        return False

    def rotate(self,observation):
        currentBlock = self.CurrentBlock

        blockDef = currentBlock.blocktypes[currentBlock.blocktype]
        rotS = currentBlock.calcNextRot()
        rotatedBlock = currentBlock.calcRotBlock(rotS, blockDef)
        absPos = currentBlock.calcAbsolutPos(rotatedBlock)
       
        allowed = True
        for absolutPosition in absPos:
            apx, apy = absolutPosition
            if apx > 9 or apx < 0 or self.inField(apx,apy) or apy > 19 or apy < 0:
                allowed = False
        
        if allowed:
            self.CurrentBlock.rotate()

    def getObservation(self):
        observation = np.zeros((20, 10), dtype=np.uint8)
        for block in self.GameField:
            x,y,col = block
            observation[y][x] = 1
        return observation

    def render(self, mode='human', close=False):
        self.screen.fill((0, 0, 0))
        for block in self.GameField:
            x,y,col = block
            pygame.draw.rect(self.screen, col,
                                 (x*30, y*30, 30, 30))
        
        for block in self.CurrentBlock.absolutPositions():
            x,y = block
            pygame.draw.rect(self.screen, self.CurrentBlock.getColor(),
                                 (x*30, y*30, 30, 30))
        pygame.display.flip()

    def newBlock(self, First = False):
        if(self.CurrentBlock):
            for pos in self.CurrentBlock.absolutPositions():
                cbx, cby = pos
                self.GameField.append((cbx,cby,self.CurrentBlock.getColor()))
        self.newBlockWithoutSave()
    
    def newBlockWithoutSave(self):
        random.seed(self.BlockCount)
        self.BlockCount += 1
        self.CurrentBlock = TetrisBlock(random.randint(0,6), 3, 0)

    def stop(self):
        pygame.quit()



