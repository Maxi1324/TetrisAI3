from Config.ConfigLoader import ConfigLoader, ConfigSpec
from Env.Env import TetrisE
from Policies.PolicyMan import PolicyMan, PolicieSpec
from models.ModelMan import NetworkSpec, NetworkMan
from Runner.Train import Trainer


config = ConfigLoader("config.ini","Main")

trainStep = int(config.getConfig(ConfigSpec.trainStep))
render = config.getConfig(ConfigSpec.render) == "True"
learningrate = float(config.getConfig(ConfigSpec.learningrate))
warmupSteps = int(config.getConfig(ConfigSpec.warmup))
limitSteps = int(config.getConfig(ConfigSpec.steplimit))
windowLength = int(config.getConfig(ConfigSpec.winlength))
targetModelUpdate = float(config.getConfig(ConfigSpec.targetModelUpdate))
verbose = int(config.getConfig(ConfigSpec.verbose))

modelSpec = NetworkSpec[config.getConfig(ConfigSpec.model)]
policySpec = PolicieSpec[config.getConfig(ConfigSpec.policy)]

minToWinLines = int(config.getConfig(ConfigSpec.minToWinLines))

env = TetrisE(render,minToWinLines)

networkMan = NetworkMan(windowLength)
model = networkMan.getNetwork(modelSpec)
print(model.summary())

policyMan = PolicyMan(minV=-1,maxV=1,testV=0,trainStep=trainStep)
policy = policyMan.getPolicy(policySpec)


trainer = Trainer(
    Env=env,
    policy=policy, 
    trainSteps=trainStep, 
    model=model,
    learningRate=learningrate, 
    warmupSteps=warmupSteps, 
    limitSteps= limitSteps,
    windowLength=windowLength, 
    targetModelUpdate=targetModelUpdate, 
    verbose=verbose 
)


trainer.train()
trainer.save(policySpec.value,modelSpec.value)





