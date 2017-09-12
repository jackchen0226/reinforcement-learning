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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        # epsilon = np.exp(-1 / number replays)
        self.epsilon_decay = 0.0998
        self.learning_rate = 0.005
        self.model = self._build_model()
        """
        variables = self.model.get_weights()
        self.weights = []
        for i, var in enumerate(variables):
            self.weights.append(K.placeholder(dtype='float32'))
        self.gradients = K.gradients(self.cartpole_loss, variables)
        self.update = 
        """
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
        return -K.mean(K.log(prediction + 0.001) * (1-r) * target, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=self.cartpole_loss,
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    #input is state and action, output is reward, just like in replay() use target[0][action]
    def replay_episode(self, state_history, reward_history, action_history, pred_next_state, done):
        reward = self.discounted_reward(reward_history, self.gamma)
        reward = (reward - np.mean(reward)) / np.std(reward)
        target_f = np.zeros((1, 2))

        debug1 = self.model.predict(state_history[0])

        for i, next_state in enumerate(pred_next_state):

            target_f[0][action_history[i]] = reward[i]

            target_f[0][1 - action_history[i]] = -1e4
            self.model.fit(state_history[i], target_f, epochs=5, verbose=0)
            
        
        debug2 = self.model.predict(state_history[0])

        print("{} + {}, {} = {}".format(debug1, action_history[0], reward[0], debug2))

    """
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        r = []
        for tup in minibatch:
            # List of rewards
            r.append(tup[2])
        r = self.discounted_reward(r, self.gamma)

        #r = (r - np.mean(r)) / np.std(r)
        
        # After discounting and normalizing rewards, add the new rewards them back into minibatch
        for i, tup in enumerate(minibatch):
            # tup[2] is the rewards in the minibatch tuple
            tup = list(tup)
            tup[2] = r[i]
            tup = tuple(tup)
            minibatch[i] = tup
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            target_f[0][1 - action] = -1e4
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    """

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
    #env = wrappers.Monitor(env, 'cartpolev0-experiment', force=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    #state_history = deque(maxlen=2000)
    #reward_history = deque(maxlen=2000)
    #action_history = deque(maxlen=2000)
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state_history = []
        reward_history = []
        action_history = []
        pred_next_state = []
        for time in range(500):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            #print(state)
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
