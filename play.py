from utils import preprocess, stack_frames, huber_loss, one_hot, NN
from agent import Pong_agent

if __name__ == "__main__":
    environment = "PongDeterministic-v4"
    agent = Pong_agent(environment)
    agent.play()