import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import gymnasium as gym
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity, observation_dim):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.observation_dim = observation_dim

    def store(self, observation, action, reward, next_observation, done):
        # One-hot encode observations
        observation = np.eye(self.observation_dim)[observation]
        next_observation = np.eye(self.observation_dim)[next_observation]
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, size):
        batch = random.sample(self.memory, size)
        observation, action, reward, next_observation, done = zip(*batch)
        return np.array(observation), action, reward, np.array(next_observation), done

    def __len__(self):
        return len(self.memory)


class QR_DQN(nn.Module):
    def __init__(self, observation_dim, action_dim, quant_num):
        super(QR_DQN, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.quant_num = quant_num

        self.fc1 = nn.Linear(self.observation_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, self.action_dim * self.quant_num)

    def forward(self, observation):
        batch_size = observation.size(0)
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(batch_size, self.action_dim, self.quant_num)
        return x

    def act(self, observation, epsilon):
        self.eval()
        observation = observation.to(device)
        if random.random() > epsilon:
            with torch.no_grad():
                dist = self.forward(observation)
                q_values = dist.mean(2)
                action = q_values.argmax(dim=1).item()
        else:
            action = random.choice(list(range(self.action_dim)))
        self.train()
        return action


def get_target_distribution(target_model, next_observation, reward, done, gamma, action_dim, quant_num):
    batch_size = next_observation.size(0)
    device = next_observation.device

    next_dist = target_model.forward(next_observation).detach()
    next_action = next_dist.mean(2).max(1)[1].detach()
    next_action = next_action.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, quant_num)
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand(batch_size, quant_num)
    done = done.unsqueeze(1).expand(batch_size, quant_num)
    target_dist = reward + gamma * (1 - done) * next_dist
    target_dist = target_dist.clamp(0, 1)

    quant_idx = torch.argsort(next_dist, dim=1)
    tau_hat = torch.linspace(0.0, 1.0 - 1. / quant_num, quant_num, device=device) + 0.5 / quant_num
    tau_hat = tau_hat.unsqueeze(0).expand(batch_size, quant_num)
    tau = tau_hat.gather(1, quant_idx)

    return target_dist, tau


def train(eval_model, target_model, buffer, optimizer, gamma, action_dim, quant_num, batch_size, count, update_freq, k=1.):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_observation = torch.FloatTensor(next_observation).to(device)
    done = torch.FloatTensor(done).to(device)

    dist = eval_model.forward(observation)
    action = action.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, quant_num)
    dist = dist.gather(1, action).squeeze(1)
    target_dist, tau = get_target_distribution(target_model, next_observation, reward, done, gamma, action_dim, quant_num)

    u = target_dist - dist

    huber_loss = 0.5 * u.abs().clamp(min=0., max=k).pow(2)
    huber_loss += k * (u.abs() - u.abs().clamp(min=0., max=k) - 0.5 * k)
    quantile_loss = (tau - (u < 0).float()).abs() * huber_loss
    loss = quantile_loss.sum() / batch_size

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(eval_model.parameters(), 0.5)
    optimizer.step()

    if count % update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


if __name__ == '__main__':
    # Hyperparameters
    epsilon_init = 0.95
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99
    learning_rate = 1e-3
    capacity = 100000
    exploration = 200
    episodes = 500
    quant_num = 10
    update_freq = 200
    batch_size = 64
    k = 1.
    render = True  # Set to False for faster training

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    env = env.unwrapped
    observation_dim = env.observation_space.n
    action_dim = env.action_space.n
    buffer = ReplayBuffer(capacity, observation_dim)

    # Initialize networks
    eval_net = QR_DQN(observation_dim, action_dim, quant_num).to(device)
    target_net = QR_DQN(observation_dim, action_dim, quant_num).to(device)
    target_net.load_state_dict(eval_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode

    # Initialize optimizer
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)

    # Initialize variables
    count = 0
    epsilon = epsilon_init
    weight_reward = None

    for i in range(episodes):
        obs, info = env.reset()
        reward_total = 0
        if render:
            env.render()
        while True:
            # Convert observation to one-hot tensor
            obs_one_hot = np.eye(observation_dim)[obs]
            obs_tensor = torch.FloatTensor(obs_one_hot).unsqueeze(0).to(device)  # Shape: [1, observation_dim]
            action = eval_net.act(obs_tensor, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.store(obs, action, reward, next_obs, done)
            if render:
                env.render()
            reward_total += reward
            count += 1
            obs = next_obs
            if len(buffer) > exploration:
                train(eval_net, target_net, buffer, optimizer, gamma, action_dim, quant_num, batch_size, count, update_freq, k=1.)
            if done:
                if epsilon > epsilon_min:
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                if weight_reward is None:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print(f'episode: {i+1}  reward: {reward_total}  weight_reward: {weight_reward:.3f}  epsilon: {epsilon:.2f}')
                break

    # Save the trained model
    torch.save(eval_net.state_dict(), 'qr_dqn_frozenlake.pth')
env.close()