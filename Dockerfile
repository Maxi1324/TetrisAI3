FROM tensorflow/tensorflow

WORKDIR /app

COPY . .

RUN pip install numpy
RUN pip install gym
RUN pip install keras-rl2==1.0.5
RUN pip install pygame

VOLUME ["models","Config"]

LABEL modeloptions="value"
LABEL policyoptions="value"

ENV model="DenseTiny"
ENV policy="Greedy"

    
CMD [ "python","src/Main.py "+${model}+" "+${policy}  ]    