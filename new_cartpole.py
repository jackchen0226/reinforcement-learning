import numpy as np
import gym
from collections import deque

# Create model
from keras.optimizers import RMSprop
from keras.layers import *
from keras.models import Sequential

class Model:
	"""docstring for Model"""
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = 0.00025

		self.model = self.create_model()

	def create_model(self):
		model = Sequential()

		model.add(Dense(64, activation='relu', input_dim=self.state_size))
		model.add(Dense(self.action_size, activation='linear'))

		model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))

		return model

	def train(self, states, target, epochs=1, verbose=0):
		self.model.fit(states, target, batch_size=64, epochs=epochs, verbose=verbose)

	def predict(self, states):
		return self.model.predict(states)

MEMORY_LEN = 100000
GAMMA = .999

EPSILON_MAX = 1.
EPSILON_MIN = 0.001
LAMBDA = 0.001

class Agent:
	"""docstring for Agent"""
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size

		self.memory = deque(maxlen=MEMORY_LEN)
		self.model = Model(state_size, action_size)

	def act(self):