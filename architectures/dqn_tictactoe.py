import math
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_epsilon(episode, epsilon_start=1.0, epsilon_final=0.6 , total_episodes=100000):

    tau = total_episodes / math.log(epsilon_start / epsilon_final)
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-episode / tau)
    if epsilon < epsilon_final:
        epsilon = epsilon_final
    return epsilon


def train_dqn(env, num_episodes=200000, gamma=0.99, learning_rate=0.001, memory_size=20000, save_interval=10000, batch_size = 64):
    env.reset()

    # Get observation space dimensions (3x3 grid with 2 planes)
    first_agent = env.possible_agents[0]
    observation_dim = env.observation_space(first_agent)["observation"].shape[0] * \
                      env.observation_space(first_agent)["observation"].shape[1] * \
                      env.observation_space(first_agent)["observation"].shape[2]
    print("Observation dimension:", observation_dim)

    action_dim = env.action_space(first_agent).n

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(observation_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=0.00001)
    criterion = nn.MSELoss()

    memory = deque(maxlen=memory_size)
    episode_rewards = {agent: [] for agent in env.possible_agents}  # Use saved possible_agents

    for episode in tqdm(range(num_episodes)):
        env.reset()
        total_rewards = {agent: 0 for agent in env.possible_agents}  # Track total rewards for each agent
        steps = 0
        last_observations = {}
        last_actions = {}
        epsilon = get_epsilon(episode)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            steps += 1

            observation_board = observation["observation"].flatten()
            total_rewards[agent] += reward

            # Store the experience from the previous turn, if available
            if agent in last_observations:
                prev_observation = last_observations[agent]
                prev_action = last_actions[agent]

                # Store experience in memory
                memory.append((prev_observation, prev_action, reward,
                               observation_board.copy(), termination))

                # Update the model
                if len(memory) > 1000:
                    batch = random.sample(memory, batch_size)
                    batch_observations, batch_actions, batch_rewards, batch_next_obs, batch_dones = zip(*batch)

                    batch_observations = torch.FloatTensor(batch_observations).to(device)
                    batch_actions = torch.LongTensor(batch_actions).to(device)
                    batch_rewards = torch.FloatTensor(batch_rewards).to(device)
                    batch_next_obs = torch.FloatTensor(batch_next_obs).to(device)
                    batch_dones = torch.FloatTensor(batch_dones).to(device)

                    # Get predicted Q-values for current states
                    q_values = model(batch_observations).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                    # Compute Q-targets for next states
                    with torch.no_grad():
                        next_q_values = model(batch_next_obs).max(1)[0]
                        q_targets = batch_rewards + (gamma * next_q_values * (1 - batch_dones))

                    loss = criterion(q_values, q_targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)

            if termination or truncation:
                action = None
            else:
                action_mask = observation["action_mask"]
                legal_actions = [i for i, mask in enumerate(action_mask) if mask]

                if not legal_actions:
                    action = None
                else:
                    if random.random() < epsilon:
                        action = random.choice(legal_actions)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(observation_board).unsqueeze(0).to(device)
                            q_values = model(state_tensor)
                            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(device)
                            q_values[0][~action_mask_tensor] = -float('inf')  # Mask illegal actions
                            action = q_values.argmax().item()

                last_observations[agent] = observation_board.copy()
                last_actions[agent] = action

            env.step(action)

        for agent in env.possible_agents:
            episode_rewards[agent].append(total_rewards[agent])

        if (episode + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'models_7_5/model_dqn_tictactoe_{ episode + 1}.pth')

        # Print episode summary
        '''print(f'Episode {episode + 1}/{num_episodes}:')
        for agent in possible_agents:
           print(f'  {agent} Total Reward: {total_rewards[agent]}')
        print(f'  Steps: {steps}')
        print(f'  Epsilon: {epsilon:.4f}')
        print()'''

    return model, episode_rewards


