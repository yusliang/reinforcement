import gym
from agent_base import AgentBase, Transition
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_probability as tfp
import random
import numpy as np


class Model(K.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = K.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = K.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = K.layers.Dense(num_actions, activation='relu')
        self.softmax = K.layers.Softmax()

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        print(x)
        x = self.softmax(x)
        print(x)
        return x

    def get_action(self, inputs):
        prob = self.predict(inputs)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        print(inputs, prob, dist)
        action = dist.sample()
        return int(action.numpy())


class Agent(AgentBase):
    def __init__(self, env, model):
        super().__init__(env, model)
        self.opt = None

    def train(self, steps, gamma=0.95, learning_rate=0.001, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
              done_punish=-10):
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10.0)  # do gradient clip
        state = self.env.reset()
        states = []
        rewards = []
        actions = []
        step = 0
        while step < steps:
            action = self.model.get_action(state[None])
            epsilon_ = max(min_epsilon, epsilon * np.power(epsilon_decay, step))
            action = self._get_action(action, epsilon_)
            print(action)
            state_next, reward, done, _ = self.env.step(action)
            step += 1
            if done:
                reward = done_punish
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            if done:
                for i in range(len(rewards) - 2, -1, -1):
                    rewards[i] = gamma * rewards[i+1]
                self.update_model(states, rewards, actions, step)
                state = self.env.reset()
                states = []
                rewards = []
                actions = []
            else:
                state = state_next
        pass

    def _get_action(self, action, epsilon=0.0):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return action

    def update_model(self, states, rewards, actions, step):
        for state, reward, action in zip(states, rewards, actions):
            with tf.GradientTape() as tape:
                prob = self.model(state[None], training=True)
                dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
                log_prob = dist.log_prob(action)
                loss = -log_prob * reward
                if step % 1000 == 0:
                    print("loss:", loss)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))


def train_and_evaluate(env, steps, gamma,
                       learning_rate, epsilon, epsilon_decay, min_epsilon, done_punish,
                       rounds=10, play_after_learn=False):
    num_actions = env.action_space.n
    reward_list = []
    agent = None
    for i in range(rounds):
        tf.keras.backend.clear_session()
        model = Model(num_actions)
        agent = Agent(env, model)
        agent.train(round(steps, 0), gamma, learning_rate, epsilon,
                    epsilon_decay, min_epsilon, done_punish)
        reward_list.append(agent.evaluation(10))
    print(steps, gamma,
          learning_rate, epsilon, epsilon_decay)
    reward = sum(reward_list) / rounds
    print("reward: {}".format(reward))
    if play_after_learn:
        agent.play()
    return reward

if '__main__' == __name__:
    env = gym.make('CartPole-v0')  # insert your favorite environment
    train_and_evaluate(env, steps=100, gamma=0.95, learning_rate=0.001, epsilon=.95, epsilon_decay=.95, min_epsilon=.01, done_punish=-100)
    # model = Model(env.action_space.n)
    # agent = Agent(env, model)
    # print("before train:", agent.evaluation(10))    # better than dqn, since action is randomly in set {0, 1}
    # agent.train(steps=100, gamma=0.95, learning_rate=0.001)
    # print("after train:", agent.evaluation(10))
    # agent.play()
