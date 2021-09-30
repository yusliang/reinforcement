import numpy as np
import time
from collections import namedtuple


Transition = namedtuple("Transition", field_names=["state", "state_next", "action", "reward", "done"])


class AgentBase:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def evaluation(self, rounds):
        # play #rounds, each round take #max_steps, calc avg_steps that model can take before done
        steps = 0
        for i in range(rounds):
            state, done, step = self.env.reset(), False, 0
            while not done:
                action = self.model.get_action(state[None])
                state, _, done, _ = self.env.step(action)
                step += 1
            steps += step
        return steps / rounds

    def play(self):
        state = self.env.reset()
        steps = 0
        while True:
            self.env.render()
            action = self.model.get_action(state[None])
            observation, reward, done, info = self.env.step(action)
            steps += 1
            time.sleep(0.05)
            if done:
                break
        print("play steps: ", steps)

    def _init_restore_buffer(self, buffer_size):
        self.buffer_size = buffer_size
        self.state_buffer = np.empty([buffer_size, self.env.reset().shape[0]], dtype=np.float32)
        self.state_next_buffer = np.empty([buffer_size, self.env.reset().shape[0]], dtype=np.float32)
        self.action_buffer = np.empty(buffer_size, dtype=np.int8)
        self.reward_buffer = np.empty(buffer_size, dtype=np.float32)
        self.done_buffer = np.empty(buffer_size, dtype=np.int8)
        self.next_buffer_index = 0
        self.num_transition = 0

    def _store_to_replay_buffer(self, transition):
        idx = self.next_buffer_index
        self.state_buffer[idx] = transition.state
        self.state_next_buffer[idx] = transition.state_next
        self.action_buffer[idx] = transition.action
        self.reward_buffer[idx] = transition.reward
        self.done_buffer[idx] = transition.done
        self.next_buffer_index = (idx + 1) % self.buffer_size
        self.num_transition = min(self.num_transition + 1, self.buffer_size)
