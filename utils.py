#import stuff
import keras
import random
import numpy as np
import cv2
from keras.layers import Input, Conv2D, Dense, Multiply
from keras.layers.core import Flatten
from keras import backend as K
from collections import deque
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense, Activation, Conv2D, Flatten, Multiply
from keras.optimizers import Adam


#Preprocess images (frames), for less heavy computations
def preprocess(img):
    #Transform to grayscale
    newimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    #From 210x160, to 84x84
    newimg = cv2.resize(newimg, (84,84))
    newimg = newimg/255.0
    return newimg

#Stack of 4 consecutive frames
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

#loss function
def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

#One hot transformation
def one_hot(batches, targets, classes):
    new_targets = []
    for i in range(batches):
        if targets[i] == 2:
            new_targets.append(0)
        else:
            new_targets.append(1)
    return np.eye(classes)[np.array(new_targets).reshape(-1)]

#Memory class for experience replay
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def remember(self, experience):
        self.buffer.append(experience)

    def minibatch(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = 32, replace = False)
        return [self.buffer[i] for i in index]

        
#Nework architechture
def NN(input_shape, action_space, lr):
    frames = Input(input_shape, name = "frames")
    actions_input = Input((action_space, ), name = "mask")
    c1 = Conv2D(32, (8,8), strides = (4,4), activation = "relu", kernel_initializer = "he_normal", name = "Conv1")(frames)
    c2 = Conv2D(64, (4,4), strides = (2,2), activation = "relu", name = "Conv2")(c1)
    c3 = Conv2D(64, (3,3), activation = "relu", name = "Conv3")(c2)
    flat = Flatten(name = "flatten")(c3)
    output = Dense(action_space, name = "output")(flat)
    filtered_output = Multiply(name = "Qvalue")([output, actions_input])
    model = Model(inputs = [frames, actions_input], outputs = filtered_output)
    optimizer = Adam(lr = lr)
    model.compile(optimizer, loss = huber_loss)
    #model.summary()
    return model