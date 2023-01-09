from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import os
from datetime import datetime
class Trainer:
    def __init__(self, Env, policy,trainSteps,model, learningRate, warmupSteps, limitSteps,windowLength,targetModelUpdate,verbose, Logger) -> None:
        self.trainSteps = trainSteps
        self.model = model
        self.policy = policy
        self.Env = Env
        self.learningRate = learningRate
        self.warmupSteps = warmupSteps
        self.limitSteps = limitSteps
        self.windowLength = windowLength
        self.targetModelUpdate = targetModelUpdate
        self.verbose = verbose
        self.Logger = Logger

        self.day = datetime.now().strftime("%Y%m%d")
        self.time = datetime.now().strftime("%H%M%S")
        self.foldername = f"models/{self.day}/{self.time}"
        
    def train(self, PolicyName = "not know", NetworkName = "not know"):
        self.Logger.Setup(self.foldername)
        self.Logger.log("Now Training for " + str(self.trainSteps)+"steps")
        self.Logger.log("Training with policy: " + PolicyName)
        self.Logger.log("Training with model: " + NetworkName)
        self.Logger.log("Training with learning rate: " + str(self.learningRate))
        self.Logger.log("Training with warmup steps: " + str(self.warmupSteps))
        self.Logger.log("Training with limit steps: " + str(self.limitSteps))
        self.Logger.log("Training with window length: " + str(self.windowLength))
        self.Logger.log("Training with target model update: " + str(self.targetModelUpdate)+"\n")
        self.Logger.log("Start Training at " + datetime.now().strftime("%Y%m%d-%H%M%S")+"\n\n")       

        memory = SequentialMemory(limit=self.limitSteps, window_length=self.windowLength)
        dqn = DQNAgent(model=self.model, nb_actions=3, memory=memory, nb_steps_warmup=self.warmupSteps,
                    target_model_update=self.targetModelUpdate, policy=self.policy)
        dqn.compile(Adam(learning_rate=self.learningRate))

        dqn.fit(self.Env, nb_steps=self.trainSteps, visualize=False, verbose=self.verbose)
    
        self.Logger.log("\n\nEnd Training at " + datetime.now().strftime("%Y%m%d-%H%M%S"))


    def load(self, network,policy,day,time):
        self.Logger.log("Loading model: " +  str(network) )
        foldername = f"models/{network}/{policy}/{day}/{time}"
        self.model.load_weights(foldername + "/model.hf5")

    def save(self, policy, network):
        self.Logger.log("Saving model: " + str(network) )
       
        foldername = self.foldername
        self.model.save_weights(foldername + "/model.hf5")
        self.Logger.log("Model saved at: " + foldername + "/model.hf5")
        return foldername