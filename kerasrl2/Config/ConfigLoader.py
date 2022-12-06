from configparser import ConfigParser
from enum import Enum
from Policies.PolicyMan import PolicieSpec
from models.ModelMan import NetworkSpec


class ConfigSpec(Enum):
    learning_rate = 'learning_rate'
    render = 'render'
    model = 'model'
    policy = 'policy'
    trainStep = 'trainStep'

default_config = {
ConfigSpec.learning_rate: "0.001",
ConfigSpec.render : "False",
ConfigSpec.model : NetworkSpec.DenseTiny.value,
ConfigSpec.policy : PolicieSpec.Greedy.value,
ConfigSpec.trainStep : "1000000"
}

class ConfigLoader:
    def __init__(self, config_file, MainSection):
        self.MainSection = MainSection
        self.initConfigFile(config_file, MainSection)
        
    def initConfigFile(self,config_file = 'config.ini', MainSection = "Main"):
        config = ConfigParser()
        config.read(config_file)

        for config_spec in ConfigSpec:
            if config_spec.value not in config:
                if not config.has_section(MainSection):
                    config.add_section(MainSection)
                if not config.has_option(MainSection, config_spec.value):
                    config.set(MainSection, config_spec.value, default_config[config_spec])

        with open(config_file, 'w') as f:
            config.write(f)
        self.config = config

    def getConfig(self,ConfigSpec:ConfigSpec):
        return self.config.get(self.MainSection,ConfigSpec.value)

