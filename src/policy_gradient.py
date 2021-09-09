import gym
from agent_base import AgentBase, Transition
import tensorflow as tf
from tensorflow import keras as K
import tensorflow_probability as tfp


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
        x = self.softmax(x)
        return x

    def get_action(self, inputs):
        prob = self.predict(inputs)
        print("prob", prob)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        print("dist", dist)
        action = dist.sample()
        print("action", action)
        return int(action.numpy())


class Agent(AgentBase):
    def __init__(self, env, model, buffer_size):
        super().__init__(env, model, buffer_size)

    def play(self):
        state = self.env.reset()
        while True:
            self.env.render()
            action = self.model.get_action(state[None])
            observation, reward, done, info = self.env.step(action)
            if done:
                break

    def train(self, steps, start_learning_steps):
        state = self.env.reset()
        for i in range(1, steps+1):
            action = self.model.get_action()
            action = self._get_action(action)
            state_next, reward, done, _ = env.step(action)
            transition = Transition(state=state, state_next=state_next, action=action,
                                    reward=reward, done=done)
            self._store_to_replay_buffer(transition)
            if steps > start_learning_steps:
                self._train_step()
            if done:
                state = self.env.reset()
            else:
                state = state_next
        pass

    def _get_action(self, action):
        return action

    def _train_step(self):

        pass


if '__main__' == __name__:
    env = gym.make('CartPole-v0')  # insert your favorite environment
    model = Model(env.action_space.n)
    agent = Agent(env, model, buffer_size=100)
    # agent.play()
