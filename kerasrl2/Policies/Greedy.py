from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def policy(minV, maxV,testV,trainStep):
    return LinearAnnealedPolicy(EpsGreedyQPolicy(), 
                              attr='eps',
                              value_max=maxV,
                              value_min=minV,
                              value_test=testV,
                              nb_steps=trainStep)