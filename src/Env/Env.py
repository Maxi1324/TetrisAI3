
import pygame
import numpy as np 
import random
from gym import Env, spaces

from Env.TetrisBlock import TetrisBlock

class TetrisE(Env):

    def __init__(self, render,MinToWinLines) -> None:
        super(TetrisE, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 10, 1), dtype=np.uint8)
        self.reward_range = (-.1, 1)
        self.doRender = render
        self.MinToWinLines = MinToWinLines

    def step(self, action):
        self.count += 1
        observation = self.getObservation()

        action = round(action)
        if action == 0:
            self.moveBlock(-1,0)
        elif action == 1:
            self.moveBlock(1,0)
        elif action == 2:
            pass
            #runter = self.moveBlock(0,1)
            #while(runter):
             #   runter = self.moveBlock(0,1)
        elif action == 3:
            self.rotate(observation)

        if self.count % 4:
            self.moveBlock(0,1)

        retObs = observation.reshape(20,10,1)
        done = self.lost()
        reward = self.calcReward(self.linedCleared,done)
        info = {}

        if(self.doRender):
            self.render()

        return retObs, reward, done, info
        
    def calcReward(self, linewCleared, done)->int:
        r = 0.00007
        if done: 
            if linewCleared >= self.MinToWinLines:
                r = 1
            else:
                r = -.1
        return r

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
        
        self.linedCleared += len(lines)
             

    def lost(self):
        for pos in self.GameField:
            x,y,col = pos
            if y == 1:
                return True
        return False

    def reset(self):
        self.linedCleared = 0
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