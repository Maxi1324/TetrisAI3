from models.DenseTiny import DenseTiny as DenseTiny
from enum import Enum

class NetworkSpec(Enum):
    DenseTiny = "DenseTiny"

Networks = {
    NetworkSpec.DenseTiny : DenseTiny
}