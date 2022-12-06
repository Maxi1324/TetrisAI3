from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import os
from datetime import datetime
class Trainer:
    def __init__(self, Env, policy,trainSteps,model, learningRate, warmupSteps, limitSteps,windowLength,targetModelUpdate,verbose) -> None:
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
        
    def train(self):
        memory = SequentialMemory(limit=self.limitSteps, window_length=self.windowLength)
        dqn = DQNAgent(model=self.model, nb_actions=5, memory=memory, nb_steps_warmup=self.warmupSteps,
                    target_model_update=self.targetModelUpdate, policy=self.policy)
        dqn.compile(Adam(learning_rate=self.learningRate))

        dqn.fit(self.Env, nb_steps=self.trainSteps, visualize=False, verbose=self.verbose)

    def load(self, network,policy,day,time):
        foldername = f"models/{network}/{policy}/{day}/{time}"
        self.model.load_weights(foldername + "/model.hf5")

    def save(self, policy, network):
        day = datetime.now().strftime("%Y%m%d")
        time = datetime.now().strftime("%H%M%S")
        foldername = f"models/{network}/{policy}/{day}/{time}"
        os.makedirs(foldername)
        self.model.save_weights(foldername + "/model.hf5")