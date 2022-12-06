from Config.ConfigLoader import ConfigLoader, ConfigSpec
from Env.Env import TetrisE
from Policies.PolicyMan import PolicyMan, PolicieSpec
from models.ModelMan import NetworkSpec, Networks



config = ConfigLoader("config.ini","Main")

trainStep = int(config.getConfig(ConfigSpec.trainStep))

render = config.getConfig(ConfigSpec.render) == "True"

#saveName format

env = TetrisE(render)
model = Networks[NetworkSpec[config.getConfig(ConfigSpec.model)]]

policyMan = PolicyMan(0,1,2,1)
policy = policyMan.getPolicy(PolicieSpec[config.getConfig(ConfigSpec.policy)])