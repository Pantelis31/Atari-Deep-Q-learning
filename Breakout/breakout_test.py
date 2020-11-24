#Import Modules------------------------------------------------------------------------
import time
import gym
import keras 
#import matplotlib.pyplot as plt
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
    newimg = cv2.resize(newimg, (84,84)) #From 210x160, to 84x84 (DeepMind)
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
env = gym.make("BreakoutDeterministic-v4")

state_size = (84,84,4) #4 channels
action_size =  3 #2 actions
learning_rate = 0.00001
batch_size = 32



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
    return model

model = NN()





model.load_weights("Breakout3_model.h5")

total_test_rewards = []
for episode in range(10):
    total_rewards = 0
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    print("EPISODE {} STARTS!".format(episode))
    while True:
        state = state.reshape(1,84,84,4)
        Qs = model.predict([state,np.ones(action_size).reshape(1,action_size)])
        max_value = np.argmax(Qs[0])
        if max_value == 0:
            action = 1
        elif max_value == 1:
            action = 2
        else:
            action = 3
        next_state, reward, done, _ = env.step(action)
        env.render(mode = "human")
        time.sleep(0.03)
        total_rewards += reward
        if done:
            print("Score: ", total_rewards)
            total_test_rewards.append(total_rewards)
            break
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state    
    env.close()



