""" Simple Q Learning Network """
import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K

from gym import wrappers

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.MEMORY = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        # epsilon = np.exp(-1 / number replays)
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00025
        self.model = self._build_model()


    def discounted_reward(self, rewards, gamma):
        ans = np.zeros_like(rewards)
        running_sum = 0
        # compute the result backward
        for i in reversed(range(len(rewards))):
            running_sum = running_sum * gamma + rewards[i]
            ans[i] = running_sum
        return ans

    def cartpole_loss(self, target, prediction):
        r = K.cast(K.less_equal(target, -1e4), K.floatx())
        # create array of target

        # If target[0] is greater than target[1], nothing changes.
        # If target [0] is less than target[1], target[0] = 0
        #target[0] *= K.cast(K.greater_equal(target[0], target[1]), K.floatx())

        return (target - prediction * (1 - r)) ** 2
        #return -K.mean(K.log(prediction + 0.001) * K.sum((1-r) * target), axis=-1)
        #good_probabilities = K.sum(prediction * (1-r))
        #log_prob = K.log(good_probabilities)
        #return -K.sum(log_prob)a


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        #model.add(keras.layers.normalization.BatchNormalization())
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.RMSprop(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #predictOne
        act_values = self.model.predict(state.reshape(1, self.state_size)).flatten()
        return np.argmax(act_values[0])  # returns action

    #input is state and action, output is reward, just like in replay() use target[0][action]
    def replay_episode(self, state_history, reward_history, action_history, next_state, done):
        #reward = self.discounted_reward(reward_history, self.gamma)
        #reward = (reward - np.mean(reward)) / np.std(reward)
        target_f = np.zeros((1, 2))

        debug1 = self.model.predict(state_history[-1])

        for i, action in enumerate(action_history):
            if done:
                target_f[0][action] = reward_history[i]
            else:
                target_f[0][action] = reward_history[i] + self.gamma * numpy.amax(self.model.predict(next_state[i]))

            #target_f[0][1 - action] = -1e4
            self.model.fit(state_history[i], target_f, epochs=1, verbose=0)
            
        
        debug2 = self.model.predict(state_history[-1])

        #print("{} + {}, {} = {}".format(debug1, action_history[-1], reward[-1], debug2))

    def epsilon_update(self):
        if self.epsilon > self.epsilon_min:
            # scale = epsilon ** number replays
            # np.exp(-1) = epsilon ** number replays
            # -1 = number replays * np.log(epsilon)
            # epsilon = np.exp(-1 / number replays)
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # RIP openai gym :(
    #env = wrappers.Monitor(env, 'cartpolev0-experiment', force=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    batch_size = 64
    agent = DQNAgent(state_size, action_size, batch_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state_history = []
        reward_history = []
        action_history = []
        pred_next_state = []
        for time in range(500):
            #env.render()
            # Agent chooses to act either randomly or the best action based on epsilon
            # Environment updates, get state and reward
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            #if done and time < 199:
            #    reward = -10
            #done and print(reward)
            next_state = np.reshape(next_state, [1, state_size])
            if done:
                next_state = None
            # Append states, rewards, and action to history
            state_history.append(state)
            reward_history.append(reward)
            action_history.append(action)
            pred_next_state.append(next_state)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
        agent.replay_episode(state_history, reward_history, action_history, pred_next_state, done)
        agent.epsilon_update()
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
