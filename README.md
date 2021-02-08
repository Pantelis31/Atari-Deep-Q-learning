## Table of Contents
- [Installation](#inst)
- [Introduction](#intro)
- [Contents](#contents)
- [Description](#desc)

## Installation
- Python 3.6.8
- Clone the repository
- Run <code>pip install -r requirements.txt</code> to install dependencies
- Run **play.py** to watch the agent play


## Introduction
This project’s goal is to explore the concept of reinforcement learning and more specifically deep Q-learning, by building two agents that are capable of playing a classic Atari game (Pong) on at least human level. It is inspired by DeepMind’s [paper](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning), where the agent learns to beat the game with no previous knowledge on it, counting on nothing but sequences of images from the actual game. It turns out that deep q-learning can be a very powerful algorithm at some cases, as the results of the agents are quite impressive, with the Pong agent managing to find a winning strategy.

## Contents
- **Pong2_model.h5** file has the model weights.
- **agent.py** contains the agent class.
- **utils.py** contains some utility functions.
- **play.py** loads the weights and plays the game.


## Description
The main goal of deep Q-learning is for the agent to learn the Q-value function, because when it actually
does, at each state it would be able to correctly evaluate the quality of any possible action and therefore act
towards beating the game. So, instead of the Q-table that we find in classic Q-learning, now there is a neural
network that tries to approximate and eventually learn the Q-value function, typically by minimizing the Mean Squared Error.

A convolutional neural network is used to receive raw frames as input and predict Q-values as output for corresponding actions within the game environment, then the action providing the maximum Q-value is being chosen by the agent. Also, the concept of experience replay is being used for the training of the agents which helps in obtaining more stable results, followed by random batch updates to avoid correlations. Overall, the training process is quite similar for the two agents, with a few differences regarding the game environments, possible actions that can be performed and rewards received. At every training step, the frames are being preprocessed and normalized in order to avoid any unnecessary computational burden. Finally, the actual input of the network is a sequence of four consecutive frames in order for the agent to be able to extract information about the current state of the game and to give a sense of motion.

The technique of experience replay is of great importance since it can improve our network’s results a lot. A transition is defined as a tuple that contains the initial state, the action performed, the reward received, the terminal state (whether the game ended or not) and the next state. Instead of learning from a single transition on every step, it is way more efficient to learn from random batches that come from the replay memory. The replay memory contains all of the last N transitions and on each step a random batch of 32 transitions is drawn from the replay memory for training. The reason for drawing random batches is because consecutive transitions are highly correlated, therefore this technique helps in breaking these correlation to a certain extend. Before the training process when the replay memory is empty, it is first initialized with a number of random transitions (the agent performs random actions).

The exploration versus exploitation dilemma has to do with how the agent acts in the environment and its also of great importance for efficient learning. The policy used for the choice of actions is called _e-greedy_. Under this policy the agent always chooses the action that yields the maximum Q-value, even at the beginning where it has no knowledge about the game at all. As one may think this could lead to serious issues where the agent makes a few actions at the beginning of the game and then gets trapped to only performing a certain action which yielded a positive reward at some point. A solution to this problem is called exploration, where at the early steps of the
training the agent performs a random action with probability _e_, this way it explores new actions or strategies and therefore doesn’t stop learning. Even though we explained the importance of exploration, it should be in balance with exploitation. Optimally, we want our agent to explore more at the early stages of the training and as it gets better and better at the game it should explore less and exploit more. Therefore we specify an exploration rate that decays over a number of steps. The exploration rate _e_ which translates as the probability of taking a random action, starts with the value of 1 and decays over a million steps finally reaching the value 0.01.



