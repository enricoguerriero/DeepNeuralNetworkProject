import torch
import torch.nn as nn
import random
from pettingzoo.classic import connect_four_v3
from pettingzoo.utils.wrappers import TerminateIllegalWrapper
import numpy as np
from dqn_connect4 import DQN

def create_env(render_mode='human'):
    env = connect_four_v3.env(render_mode=render_mode)
    env = TerminateIllegalWrapper(env, illegal_reward=-1)
    env.reset(seed=42)
    return env

def select_action(model, state, legal_actions, device='cpu'):

    if state.shape[-1] == 2 and state.shape[0] != 2:
        # State has shape (height, width, channels) -> permute to (channels, height, width)
        state = np.transpose(state, (2, 0, 1))

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Shape: (1, 2, 6, 7)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)

    # Mask illegal actions by setting Q-values to -inf
    masked_q_values = torch.full_like(q_values, float('-inf'))
    masked_q_values[legal_actions] = q_values[legal_actions]

    # Select the action with the highest Q-value
    action = torch.argmax(masked_q_values).item()
    return action


# Funzione per l'agente casuale
def random_agent(legal_actions):
    return random.choice(legal_actions)

def main():
    env = create_env(render_mode='human')
    env.reset(seed=42)

    first_agent = env.possible_agents[0]
    action_dim = env.action_space(first_agent).n

    obs_shape = env.observation_space(first_agent)["observation"].shape
    height = obs_shape[0]
    width = obs_shape[1]
    input_channels = obs_shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN(input_channels, action_dim, height, width).to(device)
    model_path = "models_connect_2/model_dqn_connect4_100000.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()


    done = False
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
                    # Model
                    action = select_action(model, state, legal_actions, device=device)
                else:
                    # Random Agent
                    #action = random_agent(legal_actions)
                    action = select_action(model, state, legal_actions, device=device)

                env.step(action)
                step += 1
            else:
                env.step(None)

    print("Game over.")

if __name__ == "__main__":
    main()