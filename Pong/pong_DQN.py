#Import Modules------------------------------------------------------------------------

import gym
import keras 
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from keras import layers
from keras import initializers
from keras import backend as K
from collections import deque
from keras.models import Model
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense, Activation, Conv2D, Flatten, Multiply
from keras.optimizers import Adam
from keras import initializers

#Preprocessing---------------------------------------------------------------------------
def preprocess(img): #Preprocess images (frames), for less heavy computations
    newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8) #Transform to grayscale
    newimg = cv2.resize(newimg, (84,84)) #From 210x160, to 84x84 
    newimg = newimg/255.0
    return newimg

stack_size = 4
stacked_frames = deque([np.zeros((84,84), dtype = np.int) for i in range(stack_size)], maxlen = 4)

#We want the NN to take as input a stack of 4 frames of the game 
def stack_frames(stacked_frames, state, new_episode):
    frame = preprocess(state)
    if new_episode:
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(4)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2) 
    return stacked_state, stacked_frames


#Initialize environment
env = gym.make("PongDeterministic-v4")
#env.get_action_meanings() 2=right, 3=left

#Hyperparameters-------------------------------------------------------------------------------
state_size = (84,84,4) #4 channels
action_size =  2 #2 actions
learning_rate = 0.00025

#Training parameters
episodes = 1000
total_steps = 20000
batch_size = 32

#Exploration and Decay rates
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = (epsilon - epsilon_min)/1000000 #decays over a million steps
gamma = 0.99 #discount parameter

#Memory parameters
init_exp = 1200 #Initial experiences 
memory_size = 10000 #Total memory size 


render_episode = False 

#Deep-Q Learning-----------------------------------------------------------------------------------    
def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss


def NN():
    frames = keras.layers.Input(state_size, name = "frames")
    actions_input = keras.layers.Input((action_size, ), name = "mask")
    c1 = keras.layers.Conv2D(32, (8,8), strides = (4,4), activation = "relu", kernel_initializer = "he_normal", name = "Conv1")(frames)
    c2 = keras.layers.Conv2D(64, (4,4), strides = (2,2), activation = "relu", name = "Conv2")(c1)
    c3 = keras.layers.Conv2D(64, (3,3), activation = "relu", name = "Conv3")(c2)
    flat = keras.layers.core.Flatten(name = "flatten")(c3)
    output = keras.layers.Dense(action_size, name = "output")(flat)
    filtered_output = keras.layers.Multiply(name = "Qvalue")([output, actions_input])
    model = keras.models.Model(input = [frames, actions_input], output = filtered_output)
    optimizer = keras.optimizers.Adam(lr = learning_rate)
    model.compile(optimizer , loss = huber_loss)
    model.summary()
    return model



model = NN()
target_model = clone_model(model)
target_model.set_weights(model.get_weights())

def one_hot(targets, classes):
    new_targets = []
    for i in range(batch_size):
        if targets[i] == 2: 
            new_targets.append(0)
        else:
            new_targets.append(1)
    return np.eye(classes)[np.array(new_targets).reshape(-1)]

def update_target():
    target_model.set_weights(model.get_weights())
    
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        
    def remember(self, experience):
        self.buffer.append(experience)
        
    def minibatch(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
        return [self.buffer[i] for i in index]
    
#Initialize memory with 1200 random experiences
memory = Memory(max_size = memory_size)
for i in range(init_exp):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    action = random.randint(2,3)
    next_state, reward, done, info = env.step(action)
    #stack frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    if done:
        next_state = np.zeros(state.shape) #restart state
        memory.remember((state, action, reward, next_state, done))
    else:
        memory.remember((state, action, reward, next_state, done))
        state = next_state


        
        

def act(epsilon, state):
        if np.random.rand() <= epsilon:
            action = random.randint(2,3)
            return action
        state = state.reshape(-1,84,84,4)
        Qs = model.predict([state, np.ones(action_size).reshape(1,action_size)])
        max_value = np.argmax(Qs[0])
        if max_value == 0:
            action = 2
        else:
            action = 3
        return action  

#TRAINING ------------------------------------------------------------------------------------------------------------------
exploration = [1]
overall_steps = 0
rewards_list = []
avg_reward = []
for episode in range(episodes):
    step = 0
    episode_rewards = []
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    while step < total_steps:
        step += 1
        overall_steps += 1
        #print("Step: {}/{}".format(step,total_steps))
        #Predict the action to take
        action = act(epsilon, state)
        next_state, reward, done, _ = env.step(action)
        if render_episode:
            env.render()
        
        #Add reward to total reward
        episode_rewards.append(reward)
        if done:
            next_state = np.zeros((84,84,3), dtype = np.uint8) #restart next state
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            #get total reward of the episode
            total_reward = np.sum(episode_rewards)
            avg_reward.append((episode, total_reward/step))
            step = total_steps
            print("EPISODE: {} SCORE: {}".format(episode, total_reward))
            rewards_list.append((episode, total_reward))
            #Store to memory
            memory.remember((state, action, reward, next_state, done))
        else:
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            memory.remember((state, action, reward, next_state, done))
            state = next_state

        #Sample a minibatch of size 32 from memory to experience replay        
        batch = memory.minibatch(batch_size)
        states = np.array([experience[0].reshape(-1,84,84,4) for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch]) 
        next_states = np.array([experience[3]for experience in batch])
        dones = np.array([experience[4] for experience in batch])
    
        actions_mask = np.ones((batch_size, action_size))
        next_states = next_states.reshape(batch_size,84,84,4)
    
        target = np.zeros((batch_size, ))
        Qs_ns = target_model.predict([next_states, actions_mask])
        
        for i in range(0, len(batch)):
            if dones[i]:
                target[i] = rewards[i]
            else:
                target[i] = rewards[i] + gamma*np.amax(Qs_ns[i])

        action_one_hot = one_hot(actions , action_size)
        target_one_hot = action_one_hot*target[:,None]

        states = states.reshape(batch_size,84,84,4)    
        model.fit([states, action_one_hot], target_one_hot, epochs = 1, batch_size = batch_size, verbose = 0)
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay
            exploration.append(epsilon)
        if overall_steps == 10000:
            print("Target network updated")
            update_target()
            overall_steps = 0

model.save_weights("Pong2_model.h5")
rewards_list = np.array(rewards_list)
exploration = np.array(exploration)
avg_reward = np.array(avg_reward)
np.save("avg_reward2_pong.npy", avg_reward)
np.save("rewards_list2_pong.npy", rewards_list)
np.save("exploration2_pong.npy", exploration)

