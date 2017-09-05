"""Deep Q Learning Network"""
import random
import gym
from gym import wrappers
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras import backend as K

EPISODES = 2000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.highest_score = {
                "indices" : deque(maxlen=32),
                "score" : deque(maxlen=32)
                }
        self.gamma = 0.98    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        # epsilon = np.exp(-1 / number replays)
        self.epsilon_decay = 0.905
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    '''
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    '''
    def cartpole_loss(self, target, prediction):
        r = K.cast(K.less_equal(target, -1e4), K.floatx())
        return -K.mean(K.log(prediction) * (1-r) * target, axis=-1)
    
    def discounted_reward(self, rewards, gamma):
    	ans = np.zeros_like(rewards)
    	running_sum = 0
    	# compute the result backward
    	for i in reversed(range(len(rewards))):
    	    running_sum = running_sum * gamma + rewards[i]
    	    ans[i] = running_sum
    	return ans

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss=self.cartpole_loss,
                      optimizer= Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        index = len(self.memory) - 1
        try:
            if max(self.highest_score["score"]) <= reward:
                self.highest_score["score"].append(reward)
                self.highest_score["indices"].append(index)
        except ValueError:
            self.highest_score["score"].append(reward)
            self.highest_score["indices"].append(index)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        """
        if random.choice((True, False)):
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = deque(maxlen=32)
            for i in self.highest_score["indices"]:
                minibatch.append(self.memory[i])
        """
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a is predicted reward of next state
                a = self.model.predict(next_state)[0]
                # t is predicted reward nof next state different model
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            # 0 -> 1
            # 1 -> 0
            # -1 -> 1
            # -(x-1) -> 1-x
            target[0][1 - action] = -1e4
            
            target[0][action] = self.discounted_reward(target[0][action], self.gamma)
            self.model.fit(state, target, epochs=1, verbose=0)

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

    def highest_score():
        x = deque(maxlen=32)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    #env = wrappers.Monitor(env, 'cartpolev1-experiment-4', force=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            #env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        agent.epsilon_update()
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
