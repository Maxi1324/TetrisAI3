from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy

def policy(minV, maxV,testV,trainStep):
    return LinearAnnealedPolicy(BoltzmannQPolicy(), 
                              attr='eps',
                              value_max=maxV,
                              value_min=minV,
                              value_test=testV,
                              nb_steps=trainStep)