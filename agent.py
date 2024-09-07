from model import Model, Trainer
import torch
from collections import deque
import random


class Agent:
    def __init__(
        self,
        lr=0.001,
        long_memory_sampled=1000,
        long_memory_size=100000,
        gamma=0.9,
        device="cpu",
    ):
        self.gamma = gamma
        self.long_memory_sampled = long_memory_sampled
        self.memory = deque(maxlen=long_memory_size)
        self.model = Model()
        self.device = device
        self.trainer = Trainer(self.model, lr=lr, gamma=self.gamma, device=self.device)

    def get_state(self, row, cur_price, cur_money, cur_count, first_price):
        x = torch.tensor(
            [
                cur_price / cur_money,
                (cur_price * cur_count) / (cur_price * cur_count + cur_money),
                cur_price / first_price,
            ]
        ).to(self.device)

        return torch.cat((row, x), 0).to(torch.float32)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_long_memory(self):
        if len(self.memory) > self.long_memory_sampled:
            mini_sample = random.sample(self.memory, self.long_memory_sampled)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states = zip(*mini_sample)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        self.trainer.train_step(states, actions, rewards, next_states)

    def train_short_memory(self, batch_size=10_000):

        len_memory = len(self.memory)

        mini_sample = [
            self.memory[i] for i in range(len_memory + 1 - batch_size, len_memory)
        ]
        states, actions, rewards, next_states = zip(*mini_sample)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        self.trainer.train_step(states, actions, rewards, next_states)

    def get_action(self, state, add_randomness=True, randomness_level=0.08):
        final_move = [0, 0, 0]
        if add_randomness and random.random() < randomness_level:
            move = random.randint(0, 2)
            final_move[move] = 1
            return final_move

        prediction = self.model(state)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move
