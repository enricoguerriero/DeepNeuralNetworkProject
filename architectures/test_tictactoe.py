import torch
import random
from pettingzoo.classic import tictactoe_v3
from dqn_tictactoe import DQN

input_size = 9 * 2
action_size = 9
model = DQN(input_size, action_size)
model.load_state_dict(torch.load("models_7_5/model_dqn_tictactoe_180000.pth"))
model.eval()

env = tictactoe_v3.env(render_mode='human')
env.reset(seed=42)


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


def main():
    env.reset()
    done = False
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
                        # Model's turn
                        action = select_action(model, state, legal_actions)
                else:
                    # Random agent's turn
                    action = random_agent(legal_actions)

                env.step(action)
                step += 1
            else:
                env.step(None)

    print("Game over.")

if __name__ == "__main__":
    main()