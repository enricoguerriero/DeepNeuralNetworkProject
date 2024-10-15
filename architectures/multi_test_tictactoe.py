import torch
import random
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from pettingzoo.classic import tictactoe_v3
from dqn_tictactoe import DQN
# Load the trained model
input_size = 9 * 2  # The size of the flattened observation (3x3x2)
action_size = 9  # Number of possible actions (9 grid positions)
model = DQN(input_size, action_size)
model.load_state_dict(torch.load("models_7_5/model_dqn_tictactoe_180000.pth"))
model.eval()  # Set the model to evaluation mode
env = tictactoe_v3.env()

def select_action(model, state, legal_actions):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)
    # Mask illegal actions
    masked_q_values = torch.full_like(q_values, float('-inf'))
    masked_q_values[legal_actions] = q_values[legal_actions]
    # Select the action with the highest Q-value
    return torch.argmax(masked_q_values).item()


def random_agent(legal_actions):
    return random.choice(legal_actions)


def play_game(env, agent_1, agent_2):
    env.reset()
    done = False
    rewards = {"player_1": 0, "player_2": 0}
    step = 0
    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if not done:
                state = observation['observation'].flatten()
                action_mask = observation['action_mask']
                legal_actions = [i for i, mask in enumerate(action_mask) if mask]

                if agent == "player_1":
                    if step < 1:
                        action = random_agent(legal_actions)
                    else:
                        action = agent_1(model, state, legal_actions)
                else:
                    if step < 1:
                        action = random_agent(legal_actions)
                    else:
                        action = agent_2(model, state, legal_actions)

                env.step(action)
            else:
                rewards[agent] += reward
                env.step(None)

            step += 1

    return rewards["player_1"], rewards["player_2"]


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