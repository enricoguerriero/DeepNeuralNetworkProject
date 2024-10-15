import torch
import torch.nn as nn
import random
from pettingzoo.classic import connect_four_v3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dqn_connect4 import DQN

env = connect_four_v3.env()
env.reset(seed=42)

first_agent = env.possible_agents[0]
action_dim = env.action_space(first_agent).n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs_shape = env.observation_space(first_agent)["observation"].shape
height = obs_shape[0]
width = obs_shape[1]
input_channels = obs_shape[2]

model = DQN(input_channels, action_dim, height, width).to(device)
model_path = "models_connect_2/model_dqn_connect4_100000.pth"

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()

def select_action(model, state, legal_actions, device='cpu'):

    if state.shape[-1] == 2 and state.shape[0] != 2:
        # State has shape (height, width, channels) -> permute to (channels, height, width)
        state = np.transpose(state, (2, 0, 1))  # Shape: (2, 6, 7)

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Shape: (1, 2, 6, 7)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)  # Shape: (7,)

    # Mask illegal actions by setting Q-values to -inf
    masked_q_values = torch.full_like(q_values, float('-inf'))
    masked_q_values[legal_actions] = q_values[legal_actions]

    # Select the action with the highest Q-value
    action = torch.argmax(masked_q_values).item()
    return action


# Funzione per l'agente casuale
def random_agent(legal_actions):
    return random.choice(legal_actions)

def play_game(env, agent_1, agent_2):
    env.reset()
    done = False
    rewards = {"player_0": 0, "player_1": 0}
    step = 0
    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                state = observation['observation']
                action_mask = observation['action_mask']
                legal_actions = [i for i, mask in enumerate(action_mask) if mask]

                if agent == "player_0":
                    if step < 4:
                        action = random_agent(legal_actions)
                    else:
                        action = agent_1(model, state, legal_actions)
                else:
                    if step < 4:
                        action = random_agent(legal_actions)
                    else:
                        action = agent_2(model, state, legal_actions)

                env.step(action)
            else:
                rewards[agent] += reward
                env.step(None)

            step += 1

    return rewards["player_0"], rewards["player_1"]


def simulate_games(num_games, agent_1, agent_2):
    wins, losses, ties = 0, 0, 0

    for _ in tqdm(range(num_games)):
        reward_1, reward_2 = play_game(env, agent_1, agent_2)

        if reward_1 == 1:
            wins += 1
        elif reward_1 == -1:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties


def main():
    num_games = 500

    # Model vs Random Agent
    wins_random_1, losses_random_1, ties_random_1 = simulate_games(num_games,
                                                             lambda model, state, legal_actions: select_action(model, state,
                                                                                                         legal_actions) ,
                                                             lambda model, state, legal_actions: random_agent(legal_actions))

    # Random Agent vs Model
    wins_random_2, losses_random_2, ties_random_2 = simulate_games(num_games,
                                                       lambda model, state, legal_actions: random_agent(legal_actions),
                                                       lambda model, state, legal_actions: select_action(model, state,
                                                                                                         legal_actions))
    wins_self, losses_self, ties_self = simulate_games(num_games,
                                                       lambda model, state, legal_actions: select_action(model, state,
                                                                                                         legal_actions),
                                                       lambda model, state, legal_actions: select_action(model, state,
                                                                                                         legal_actions))

    # Plotting the results
    labels = ['Wins', 'Losses', 'Ties']
    results_random_1 = [wins_random_1, losses_random_1, ties_random_1]
    results_random_2 = [wins_random_2, losses_random_2, ties_random_2]
    results_self = [wins_self, losses_self, ties_self]

    x = range(len(labels))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    bars=plt.bar(x, results_random_1, color=['green', 'red', 'blue'])
    plt.xticks(x, labels)
    plt.title("Model vs Random Agent")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    plt.subplot(1, 3, 2)
    bars=plt.bar(x, results_random_2, color=['green', 'red', 'blue'])
    plt.xticks(x, labels)
    plt.title("Random Agent vs Model")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    plt.subplot(1, 3, 3)
    bars=plt.bar(x, results_self, color=['green', 'red', 'blue'])
    plt.xticks(x, labels)
    plt.title("Model vs Model")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()