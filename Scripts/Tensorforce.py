from Env import TetrisE;
from tensorforce.environments import Environment
from tensorforce.agents import  Agent
from tensorforce.execution import Runner

e = TetrisE()
Env = Environment.create(
    environment=e, max_episode_timesteps=10000,
)

agent = Agent.create(
    agent='ppo', environment=Env, batch_size=10, learning_rate=1e-3,
)

runner = Runner(
    agent=agent,
    environment=Env
)

def call(d, r):
    ep = d.episodes-1
    print("Episode: {}  Time: {}".format(ep, d.episode_agent_seconds[ep]))

runner.run(num_episodes=1000000, evaluation=True,callback=call,callback_episode_frequency=100)
runner.close()

agent.save(directory='model', format='numpy')
