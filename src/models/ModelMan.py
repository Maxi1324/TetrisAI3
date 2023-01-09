from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

from models.DenseTiny import model as DenseTiny
from models.DeneSmall import model as DenseSmall
from models.DenseBig import model as DenseBig
from models.Conv1 import model as Conv1
from models.Conv2 import model as Conv2
from models.Conv3 import model as Conv3
from models.RobertsVorschlag import model as RobertsVorschlag
from enum import Enum

class NetworkSpec(Enum):
    DenseTiny = "DenseTiny",
    DenseSmall = "DenseSmall",
    DenseBig = "DenseBig",
    Conv1 = "Conv1",
    Conv2 = "Conv2",
    Conv3 = "Conv3",
    RobertsVorschlag = "RobertsVorschlag"
    
class NetworkMan:
    def __init__(self, windowLength):
        self.windowLength = windowLength
    
    def getNetwork(self, Network: NetworkSpec):
        if Network == NetworkSpec.DenseTiny:
            return DenseTiny(self.windowLength)
        elif Network == NetworkSpec.DenseSmall:
            return DenseSmall(self.windowLength)
        elif Network == NetworkSpec.DenseBig:
            return DenseBig(self.windowLength)
        elif Network == NetworkSpec.Conv1:
            return Conv1(self.windowLength)
        elif Network == NetworkSpec.Conv2:
            return Conv2(self.windowLength)
        elif Network == NetworkSpec.Conv3:
            return Conv3(self.windowLength)
        elif Network == NetworkSpec.RobertsVorschlag:
            return RobertsVorschlag(self.windowLength)
    