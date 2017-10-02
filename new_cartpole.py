import numpy as np
import gym
from collections import deque

# Create model
from keras.optimizers import RMSprop
from keras.layers import *
from keras.models import Sequential

class Model:
	
	# All model stuff and networking goes here

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

class Memory:
	"""docstring for memory"""

	samples = dequq(maxlen=MEMORY_LEN)

	def __init__(self):
		super(memory, self).__init__()

	def add(self, sample):
		self.samples.append(sample)

	def sample(self, n)
	# returns a random sample of batch size n to be replayed
		n = min(n, len(self.samples))
		return random.sample(self.samples, n)
		
		
MEMORY_LEN = 100000
BATCH_SIZE = 64

GAMMA = .999

EPSILON_MAX = 1.
EPSILON_MIN = 0.001
LAMBDA = 0.001

class Agent:
	# Needs Act, Remember, and replay
	# variables for epsilon update
	steps = 0
	epsilon = EPSILON_MAX

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size

		self.memory = Memory(MEMORY_LEN)
		self.model = Model(state_size, action_size)

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np.random.randint(0, self.action_size-1)
		else:
			return nump.argmax(self.model.predict(state))

	def observe(self, mem_input):
		self.memory.add(mem_input)

	def epsilon_update(self):
		self.steps += 1
		self.epsilon = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * np.exp(-LAMBDA * self.steps)

	def replay(self):
		batch = self.memory.sample(BATCH_SIZE)
		batchLen = len(batch)

		# for the last state_ which DNE
		no_state = np.zeros(self.state_size)

		states = np.array([o[0] for o in batch ])
		# o[3] is next state so if o[3] DNE (last state) then have an array of zeros in states_
		states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

		predict = self.model.predict(states)
		predict_ = self.model.predict(states_)

		x = np.zeros((batchLen, self.state_size))
		y = np.zeros((batchLen, self.action_size))

		

