from enum import Enum
import Policies.Greedy as Greedy
import Policies.Boltz as Boltz

class PolicieSpec(Enum):
    Greedy = "Greedy"
    Boltz = "Boltz"
    

class PolicyMan:

    def __init__(self, minV, maxV,testV,trainStep):
        self._minV = minV
        self._maxV = maxV
        self._testV = testV
        self._trainStep = trainStep

    def getPolicy(self,policyTyp):
        if policyTyp == PolicieSpec.Greedy:
            return Greedy.policy(self._minV, self._maxV,self._testV,self._trainStep)
        elif policyTyp == PolicieSpec.Boltz:
            return Boltz.policy(self._minV, self._maxV,self._testV,self._trainStep)