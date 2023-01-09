import sys

from Config.ConfigLoader import ConfigLoader, ConfigSpec
from Env.Env import TetrisE
from Policies.PolicyMan import PolicyMan, PolicieSpec
from models.ModelMan import NetworkSpec, NetworkMan
from Runner.Train import Trainer
from Logger.FileLogger import FileLogger


config = ConfigLoader("Config/config.ini","Main")

trainStep = int(config.getConfig(ConfigSpec.trainStep))
render = config.getConfig(ConfigSpec.render) == "True"
learningrate = float(config.getConfig(ConfigSpec.learningrate))
warmupSteps = int(config.getConfig(ConfigSpec.warmup))
limitSteps = int(config.getConfig(ConfigSpec.steplimit))
windowLength = int(config.getConfig(ConfigSpec.winlength))
targetModelUpdate = float(config.getConfig(ConfigSpec.targetModelUpdate))
verbose = int(config.getConfig(ConfigSpec.verbose))

modelN= config.getConfig(ConfigSpec.model)
policyN = config.getConfig(ConfigSpec.policy)


if len(sys.argv) == 3:
    modelN = sys.argv[1]
    policyN = sys.argv[2]
    print("Using model: " + modelN + " and policy: " + policyN)

modelSpec = NetworkSpec[modelN]
policySpec = PolicieSpec[policyN]

minToWinLines = int(config.getConfig(ConfigSpec.minToWinLines))

Logger = FileLogger()

env = TetrisE(render,minToWinLines, Logger = Logger)

networkMan = NetworkMan(windowLength)
model = networkMan.getNetwork(modelSpec)
print(model.summary())

policyMan = PolicyMan(minV=0,maxV=1,testV=.05,trainStep=trainStep)
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
    verbose=verbose,
    Logger = Logger
)

trainer.train(policySpec.value,modelSpec.value[0])
savePath = trainer.save(policySpec.value,modelSpec.value[0])
Logger.save(savePath)






