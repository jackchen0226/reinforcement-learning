"""Algorithm created by Irlyue"""
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

#from gym import wrappers


def discounted_reward(rewards, gamma):
    """Compute the discounted reward."""
    ans = np.zeros_like(rewards)
    running_sum = 0
    # compute the result backward
    for i in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[i]
        ans[i] = running_sum
    return ans


def test():
    """Just a test function to make sure I'm coding the
    right thing. """
    rewards = np.array([4, 2, 2, 1])
    print(discounted_reward(rewards, 1))
    # print out some help information about the environment
    env = gym.make('CartPole-v0')
    s = env.reset()
    print('Start state: ', s)
    print('Action space: ', env.action_space.n)


class Agent(object):
    def __init__(self, input_size=4, hidden_size=2, gamma=0.95,
                 action_size=2, alpha=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.action_size = action_size
        self.alpha = alpha
        # save the hyper parameters
        self.params = self.__dict__.copy()
        # placeholders
        self.input_pl = tf.placeholder(tf.float32, [None, input_size])
        self.action_pl = tf.placeholder(tf.int32, [None])
        self.reward_pl = tf.placeholder(tf.float32, [None])
        # a two-layer fully connected network
        hidden_layer = layers.fully_connected(self.input_pl,
                                              hidden_size,
                                              biases_initializer=None,
                                              activation_fn=tf.nn.relu)
        self.output = layers.fully_connected(hidden_layer,
                                             action_size,
                                             biases_initializer=None,
                                             activation_fn=tf.nn.softmax)
        # responsible output
        one_hot = tf.one_hot(self.action_pl, action_size)
        responsible_output = tf.reduce_sum(self.output * one_hot, axis=1)
        self.loss = -tf.reduce_mean(tf.log(responsible_output) * self.reward_pl)
        # training variables
        variables = tf.trainable_variables()
        self.variable_pls = []
        for i, var in enumerate(variables):
            self.variable_pls.append(tf.placeholder(tf.float32))
        self.gradients = tf.gradients(self.loss, variables)
        solver = tf.train.AdamOptimizer(learning_rate=alpha)
        self.update = solver.apply_gradients(zip(self.variable_pls, variables))

    def next_action(self, sess, feed_dict, greedy=False):
        """Pick an action based on the current state.
        Args:
        - sess: a tensorflow session
        - feed_dict: parameter for sess.run()
        - greedy: boolean, whether to take action greedily
        Return:
            Integer, action to be taken.
        """
        ans = sess.run(self.output, feed_dict=feed_dict)[0]
        if greedy:
            return ans.argmax()
        else:
            return np.random.choice(range(self.action_size), p=ans)

    def show_parameters(self):
        """Helper function to show the hyper parameters."""
        for key, value in self.params.items():
            print(key, '=', value)


def train():
    render = True
    update_every = 3
    print_every = 50
    n_episodes = 500
    rate = 0.01
    running_reward = 0.0
    tf.reset_default_graph()
    agent = Agent(hidden_size=10, alpha=1e-1, gamma=0.95)
    agent.show_parameters()
    env = gym.make('CartPole-v0')
    #env = wrappers.Monitor(env, 'tmp/trial/', force=True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        grad_buffer = sess.run(tf.trainable_variables())
        for idx in range(len(grad_buffer)):
            grad_buffer[idx] *= 0
        for i in range(n_episodes):
            s = env.reset()
            state_history = []
            reward_history = []
            action_history = []
            current_reward = 0
            while True:
                feed_dict = {agent.input_pl: [s]}
                greedy = False
                action = agent.next_action(sess, feed_dict, greedy=greedy)
                snext, r, done, _ = env.step(action)
                if render and i % 50 == 0:
                    env.render()
                current_reward += r
                state_history.append(s)
                reward_history.append(r)
                action_history.append(action)
                s = snext
                if done:
                    state_history = np.array(state_history)
                    rewards = discounted_reward(reward_history, agent.gamma)
                    # normalizing the reward really helps
                    rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                    feed_dict = {
                        agent.reward_pl: rewards,
                        agent.action_pl: action_history,
                        agent.input_pl: state_history
                    }
                    episode_gradients = sess.run(agent.gradients,
                                                 feed_dict=feed_dict)
                    for idx, grad in enumerate(episode_gradients):
                        grad_buffer[idx] += grad

                    if i % update_every == 0:
                        feed_dict = dict(zip(agent.variable_pls, grad_buffer))
                        sess.run(agent.update, feed_dict=feed_dict)
                        # reset the buffer to zero
                        for idx in range(len(grad_buffer)):
                            grad_buffer[idx] *= 0
                    if i % print_every == 0:
                        print('episode %d, current_reward %d, running_reward %d' % (i, current_reward, running_reward))
                    break
            running_reward = rate * current_reward + (1 - rate) * running_reward


if __name__ == '__main__':
    test()
    train()