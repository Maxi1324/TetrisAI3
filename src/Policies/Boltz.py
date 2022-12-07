from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy

def policy(minV, maxV,testV,trainStep):
    return BoltzmannQPolicy()