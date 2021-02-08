#import stuff
import gym
import time
import random
import numpy as np
from collections import deque
from keras.models import clone_model
from utils import preprocess, stack_frames, huber_loss, one_hot, NN

class Pong_agent:
    #initialize
    def __init__(self, env_name):
        #Environment name
        self.env_name = env_name
        self.env = gym.make(env_name)
        #Hyperparameters
        self.state_size = (84,84,4)
        self.action_size =  2
        self.learning_rate = 0.00025
        #training parameters
        self.episodes = 1000
        self.total_steps = 20000
        self.batch_size = 32
        #Exploration and Decay rates
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        #decays over a million steps
        self.epsilon_decay = (self.epsilon - self.epsilon_min)/1000000
        self.gamma = 0.99 #discount parameter
        #Memory parameters
        self.init_exp = 1200 #Initial experiences
        self.memory_size = 10000 #Total memory size
        #stacked frames
        self.stack_size = 4
        #Initialize main and target networks
        self.model = NN(input_shape = self.state_size, action_space = self.action_size, lr = self.learning_rate)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    #Update target network
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    #Perform an action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randint(2,3)
            return action
        state = state.reshape(-1,84,84,4)
        Qs = self.model.predict([state, np.ones(self.action_size).reshape(1,self.action_size)])
        max_value = np.argmax(Qs[0])
        if max_value == 0:
            action = 2
        else:
            action = 3
        return action

    #Training using experience replay
    def train(self):
        render_episode = False
        stacked_frames = deque([np.zeros((84,84), dtype = np.int) for i in range(self.stack_size)], maxlen = 4)
        #Initialize memory size
        memory = Memory(max_size = self.memory_size)
        #Initialize memory with 1200 random experiences
        for i in range(init_exp):
            if i == 0:
                state = self.env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)
            #Perform random action
            action = random.randint(2,3)
            next_state, reward, done, info = self.env.step(action)
            #stack frames
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            if done:
                next_state = np.zeros(state.shape) #restart state
                memory.remember((state, action, reward, next_state, done))
            else:
                memory.remember((state, action, reward, next_state, done))
                state = next_state

        exploration = [1]
        overall_steps = 0
        rewards_list = []
        avg_reward = []
        for episode in range(self.episodes):
            step = 0
            episode_rewards = []
            state = self.env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            while step < self.total_steps:
                step += 1
                overall_steps += 1
                #print("Step: {}/{}".format(step,total_steps))
                #Predict the action to take
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                if render_episode:
                    self.env.render()

                #Add reward to total reward
                episode_rewards.append(reward)
                if done:
                    next_state = np.zeros((84,84,3), dtype = np.uint8) #restart next state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    #get total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    avg_reward.append((episode, total_reward/step))
                    step = self.total_steps
                    print("EPISODE: {} SCORE: {}".format(episode, total_reward))
                    rewards_list.append((episode, total_reward))
                    #Store to memory
                    memory.remember((state, action, reward, next_state, done))
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.remember((state, action, reward, next_state, done))
                    state = next_state

                #Sample a minibatch of size 32 from memory to experience replay
                batch = memory.minibatch(self.batch_size)
                states = np.array([experience[0].reshape(-1,84,84,4) for experience in batch])
                actions = np.array([experience[1] for experience in batch])
                rewards = np.array([experience[2] for experience in batch])
                next_states = np.array([experience[3]for experience in batch])
                dones = np.array([experience[4] for experience in batch])

                actions_mask = np.ones((self.batch_size, self.action_size))
                next_states = next_states.reshape(self.batch_size,84,84,4)

                target = np.zeros((self.batch_size, ))
                Qs_ns = self.target_model.predict([next_states, actions_mask])

                for i in range(0, len(batch)):
                    if dones[i]:
                        target[i] = rewards[i]
                    else:
                        target[i] = rewards[i] + self.gamma*np.amax(Qs_ns[i])

                action_one_hot = one_hot(batches = self.batch_size, targets = actions , classes = self.action_size)
                target_one_hot = action_one_hot*target[:,None]

                states = states.reshape(self.batch_size,84,84,4)
                self.model.fit([states, action_one_hot], target_one_hot, epochs = 1, batch_size = self.batch_size, verbose = 0)
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay
                    exploration.append(self.epsilon)
                if overall_steps == 10000:
                    print("Target network updated")
                    self.update_target()
                    overall_steps = 0

    #Test the agent
    def play(self):
        self.model.load_weights("Pong2_model.h5")
        stacked_frames = deque([np.zeros((84,84), dtype = np.int) for i in range(self.stack_size)], maxlen = 4)
        total_test_rewards = []
        for episode in range(10):
            total_rewards = 0
            state = self.env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            print("EPISODE {} STARTS!".format(episode))
            while True:
                state = state.reshape(1,84,84,4)
                Qs = self.model.predict([state,np.ones(self.action_size).reshape(1,self.action_size)])
                max_value = np.argmax(Qs[0])
                if max_value == 0:
                    action = 2
                else:
                    action = 3
                next_state, reward, done, _ = self.env.step(action)
                time.sleep(0.03)
                self.env.render(mode = "human")
                total_rewards += reward
                if done:
                    print("Score: ", total_rewards)
                    total_test_rewards.append(total_rewards)
                    break
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
            self.env.close()