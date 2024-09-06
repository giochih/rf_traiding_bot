import torch.nn as nn
import torch.nn.functional as torch_functional
import torch.optim as optim
import torch

class Model(nn.Module):
    def __init__(self, input_size=317):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 3)

    def forward(self, x):
        x = torch_functional.relu(self.linear1(x))
        x = torch_functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Trainer:
    def __init__(self, model, lr=0.001, gamma=0.9, device = 'cpu'):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, device = 'cpu'):
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

        prediction = self.model(state)

        target = prediction.clone()
        for idx in range(len(reward)):
            q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
