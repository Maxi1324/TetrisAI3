import os
from datetime import datetime

class FileLogger:

    def __init__(self) -> None:
        self.Messages = []

    def log(self, message):
        self.f.write(message + "\n")

    def Setup(self, path):
        os.makedirs(path)
        self.f = open(path + "/log.txt", "w")

    
    def save(self,path):
        pass