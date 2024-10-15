import torch
import random
from pettingzoo.classic import chess_v6
from pettingzoo.utils.wrappers import TerminateIllegalWrapper
import numpy as np
from dqn_chess import DQN

def create_env(render_mode='human'):
    env = chess_v6.env(render_mode=render_mode)
    env = TerminateIllegalWrapper(env, illegal_reward=-1)
    env.reset(seed=42)
    return env


def select_action(model, state, legal_actions, device='cpu'):

    # Ensure the state is in (channels, height, width) format with 111 channels
    if state.shape != (111, 8, 8):
        state = np.transpose(state, (2, 0, 1))  # Transpose to (channels, height, width)
        if state.shape[-1] == 8 and state.shape[0] != 111:
            raise ValueError(f"Expected 111 channels, but got {state.shape[0]} channels.")

    state = state.copy()  # Fix for negative strides

    # Convert the state to a tensor and add batch dimension
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Shape: (1, channels, height, width)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0)  # Get Q-values

    # Mask illegal actions by setting Q-values to -inf
    masked_q_values = torch.full_like(q_values, float('-inf'))
    masked_q_values[legal_actions] = q_values[legal_actions]

    action = torch.argmax(masked_q_values).item()
    return action

def random_agent(legal_actions):
    return random.choice(legal_actions)

def main():
    env = create_env(render_mode='human')
    env.reset(seed=42)

    first_agent = env.possible_agents[0]
    action_dim = env.action_space(first_agent).n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observation_dim = env.observation_space(first_agent)["observation"].shape[0] * \
                        env.observation_space(first_agent)["observation"].shape[1] * \
                        env.observation_space(first_agent)["observation"].shape[2]
    print("Observation dimension:", observation_dim)

    model = DQN(observation_dim, action_dim).to(device)
    model_path = "models_chess/model_dqn_chess_100000.pth"  # Sostituisci con il percorso corretto
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

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
                    # Random agent
                    action = random.choice(legal_actions)

                env.step(action)
                step += 1
            else:
                env.step(None)

    print("Game over. Total steps:", step)


if __name__ == "__main__":
    main()
