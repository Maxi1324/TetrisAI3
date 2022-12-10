import os
from datetime import datetime

class FileLogger:

    def __init__(self) -> None:
        self.Messages = []

    def log(self, message):
        self.Messages.append(message)
    
    def save(self,path):
        path+="/logs/"
        os.makedirs(path)
        with open(path+"log.txt", "w") as f:
            for s in self.Messages:
                f.write(s + "\n")