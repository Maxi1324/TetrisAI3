import numpy as np

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

