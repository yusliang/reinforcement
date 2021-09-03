import tensorflow as tf
print(tf.__version__)

import gym
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple

np.random.seed(1)
tf.random.set_seed(1)

def test_model():
    pass


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        # inputs: [batch, 4]
        # print("inputs: {}".format(inputs))
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    def get_action_value(self, observation):
        q_values = self.predict(observation)
        best_action = np.argmax(q_values, axis=-1)
        return best_action, np.max(q_values, axis=-1)


Transition = namedtuple("Transition", field_names=["state", "state_next", "action", "reward", "done"])


class DQNAgent:
    def __init__(self, model, target_model, env, buffer_size):
        self.model = model
        self.target_model = target_model
        self.env = env
        self.buffer_size = buffer_size
        self.state_buffer = np.empty([buffer_size, self.env.reset().shape[0]], dtype=np.float32)
        self.state_next_buffer = np.empty([buffer_size, self.env.reset().shape[0]], dtype=np.float32)
        self.action_buffer = np.empty(buffer_size, dtype=np.int8)
        self.reward_buffer = np.empty(buffer_size, dtype=np.float32)
        self.done_buffer = np.empty(buffer_size, dtype=np.int8)
        self.next_buffer_index = 0
        self.num_transition = 0

    def evaluation(self, rounds):
        # play #rounds, each round take #max_steps, calc avg_steps that model can take before done
        steps = 0
        for i in range(rounds):
            state, done, step = self.env.reset(), False, 0
            while not done:
                action, _ = self.model.get_action_value(state[None])
                state, _, done, _ = self.env.step(action[0])
                step += 1
            steps += step
        return steps / rounds

    def evalation1(self, render=True):
        obs, done, ep_reward = self.env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.get_action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = self.env.step(action[0])
            ep_reward += reward
            if render:  # visually show
                self.env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    def train(self, steps, batch_size, gamma, target_update_iter,
              learning_rate, epsilon, epsilon_decay, min_epsilon, start_learning_steps):
        # using "compile" cause slow prediction
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10.0)  # do gradient clip
        state = self.env.reset()
        for step in range(1, steps+1):
            # play one time
            action, value = self.model.get_action_value(state[None])
            epsilon_ = max(min_epsilon, epsilon * np.power(epsilon_decay, step))
            action = self._get_action(action[0], epsilon_)
            next_state, reward, done, _ = self.env.step(action)
            # store to buffer
            self._store_to_replay_buffer(Transition(state, next_state, action, reward, done))
            if step > start_learning_steps:
                # make one train step
                loss = self._train_step(opt, batch_size, gamma)
                if step % 1000 == 0:
                    print("step={}, loss={}".format(step, loss))

            if step % target_update_iter == 0:
                self._update_target_model()

            if done:
                state = self.env.reset()
            else:
                state = next_state

    def _get_action(self, action, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return action

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _train_step(self, opt, batch_size, gamma):
        # take a batch from buffer
        if batch_size > self.num_transition:
            sample_indices = np.arange(self.num_transition)
            np.random.shuffle(sample_indices)
        else:
            sample_indices = np.random.randint(0, self.num_transition, batch_size)

        batch_state = self.state_buffer[sample_indices]
        batch_state_next = self.state_next_buffer[sample_indices]
        batch_action = self.action_buffer[sample_indices]
        reward_batch = self.reward_buffer[sample_indices]
        done_batch = self.done_buffer[sample_indices]

        # next line should not be in tape：ValueError: Could not pack sequence.
        _, q_state_next = self.target_model.get_action_value(batch_state_next)
        with tf.GradientTape() as tape:
            q_state_indices = np.c_[np.arange(batch_action.shape[-1]), batch_action]
            q_state = tf.gather_nd(self.model(batch_state), q_state_indices)
            # if is not done, y = 1 + gamma * q_state_next
            # if is done, y = -99
            y = reward_batch - 100 * done_batch + gamma * q_state_next * (1 - done_batch)
            loss = tf.keras.losses.mean_squared_error(y, q_state)
            # calculate the gradients using our tape and then update the
            # model weights
            grads = tape.gradient(loss, self.model.trainable_variables)
            opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

    def _store_to_replay_buffer(self, transition):
        idx = self.next_buffer_index
        self.state_buffer[idx] = transition.state
        self.state_next_buffer[idx] = transition.state_next
        self.action_buffer[idx] = transition.action
        self.reward_buffer[idx] = transition.reward
        self.done_buffer[idx] = transition.done
        self.next_buffer_index = (idx + 1) % self.buffer_size
        self.num_transition = min(self.num_transition + 1, self.buffer_size)


def train_and_evaluate(env, buffer_size, steps, batch_size, gamma, target_update_iter,
                       learning_rate, epsilon, epsilon_decay, min_epsilon, start_learning_steps):
    num_actions = env.action_space.n
    rounds, reward_sum = 10, 0
    for i in range(rounds):
        tf.keras.backend.clear_session()
        model = Model(num_actions)
        target_model = Model(num_actions)
        agent = DQNAgent(model, target_model, env, buffer_size=buffer_size)
        agent.train(steps, batch_size, gamma, target_update_iter, learning_rate, epsilon,
                    epsilon_decay, min_epsilon, start_learning_steps)
        reward = agent.evaluation(10)
        print(reward)
        reward_sum += reward
    print(buffer_size, steps, batch_size, gamma, target_update_iter,
          learning_rate, epsilon, epsilon_decay, start_learning_steps)
    print("avg {} steps".format(reward_sum/rounds))


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    train_and_evaluate(env, buffer_size=100, steps=5000, batch_size=5, gamma=0.99, target_update_iter=400,
                       learning_rate=0.0015, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
                       start_learning_steps=100)
