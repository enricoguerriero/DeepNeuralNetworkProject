import os
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
import wandb
from tqdm import trange

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BoardRepresentation:
    @staticmethod
    def board_to_tensor(board):
        piece_planes = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        tensor = np.zeros((17, 8, 8), dtype=np.float32)  # Updated to 17 planes

        # 1. Piece-Type Planes (12 Planes)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                plane = piece_planes[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane += 6  # Offset for black pieces
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                tensor[plane, row, col] = 1

        # 2. Player to Move (1 Plane)
        tensor[12, :, :] = 1 if board.turn == chess.WHITE else 0

        # 3. Castling Rights (4 Planes)
        tensor[13, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0  # White Kingside
        tensor[14, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0  # White Queenside
        tensor[15, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0  # Black Kingside
        tensor[16, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0  # Black Queenside

        return tensor

class ActionEncoder:
    @staticmethod
    def encode_action(move):
        from_square = move.from_square
        to_square = move.to_square
        promotion = move.promotion if move.promotion else 0  # 0 means no promotion
        return from_square * 64 * 7 + to_square * 7 + promotion

    @staticmethod
    def decode_action(index):
        from_square = index // (64 * 7)
        to_square = (index % (64 * 7)) // 7
        promotion = (index % (64 * 7)) % 7
        promotion = promotion if promotion != 0 else None
        return chess.Move(from_square, to_square, promotion=promotion)

ACTION_SIZE = 64 * 64 * 7

class ChessEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = chess.Board()
        self.done = False
        self.previous_material = self.calculate_material_balance()
        self.previous_mobility = self.calculate_mobility()
        return self.get_state()

    def get_state(self):
        return BoardRepresentation.board_to_tensor(self.board)

    def step(self, action_idx):
        move = ActionEncoder.decode_action(action_idx)
        if move in self.board.legal_moves:
            self.board.push(move)
            reward = self.get_reward()
            done = self.board.is_game_over()
            next_state = self.get_state()
            return next_state, reward, done, {}
        else:
            # Illegal move
            reward = -10  # Penalty for illegal move
            done = True
            next_state = self.get_state()
            return next_state, reward, done, {}

    def get_reward(self):
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -100  # Loss
            else:
                return 100   # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_draw():
            return 0  # Draw
        else:
            # Intermediate rewards
            current_material = self.calculate_material_balance()
            material_reward = current_material - self.previous_material
            self.previous_material = current_material

            current_mobility = self.calculate_mobility()
            mobility_reward = current_mobility - self.previous_mobility
            self.previous_mobility = current_mobility

            repetition_penalty = -5 if self.board.is_repetition() else 0

            return material_reward + 0.1 * mobility_reward + repetition_penalty

    def calculate_material_balance(self):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        white_material = sum(
            piece_values.get(piece.piece_type, 0) for piece in self.board.piece_map().values() if piece.color == chess.WHITE
        )
        black_material = sum(
            piece_values.get(piece.piece_type, 0) for piece in self.board.piece_map().values() if piece.color == chess.BLACK
        )
        return black_material - white_material

    def calculate_mobility(self):
        white_mobility = len(list(self.board.legal_moves)) if self.board.turn == chess.WHITE else 0
        black_mobility = len(list(self.board.legal_moves)) if self.board.turn == chess.BLACK else 0
        return black_mobility - white_mobility

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ChessAgent:
    def __init__(self, action_size, model_path, env, device):
        self.action_size = action_size
        self.model_path = model_path
        self.env = env
        self.device = device
        self.policy_net = DQN(action_size).to(device)
        self.target_net = DQN(action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(1000)
        self.steps_done = 0
        self.epsilon = 1.0
        self.episode_start = 0
        self.episode_rewards = []
        self.win_rates = []
        self.losses = []
        self.wins = 0
        self.draws = 0
        self.losses_count = 0

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_start = checkpoint['episode'] + 1
            self.epsilon = checkpoint['epsilon']
            self.memory = checkpoint['memory']
            self.losses = checkpoint['losses']
            self.episode_rewards = checkpoint['episode_rewards']
            self.win_rates = checkpoint['win_rates']
            self.steps_done = checkpoint['steps_done']
            print(f"Resuming training from episode {self.episode_start}")
        else:
            print("No existing model found. Starting training from scratch.")
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

    def select_action(self, state, board):
        sample = random.random()
        legal_moves = list(board.legal_moves)
        legal_move_indices = [ActionEncoder.encode_action(move) for move in legal_moves]

        if sample < self.epsilon:
            action_idx = random.choice(legal_move_indices)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                q_values = q_values.cpu().detach().numpy()[0]
                mask = np.full(ACTION_SIZE, -np.inf)
                mask[legal_move_indices] = q_values[legal_move_indices]
                action_idx = np.argmax(mask)
        return action_idx

    def train(self, num_episodes=1500, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.4, epsilon_decay=1000000, learning_rate=1e-4, memory_capacity=1000, target_update=1000):
        self.memory = ReplayMemory(memory_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.epsilon = epsilon_start

        for episode in trange(self.episode_start, num_episodes, desc='Training'):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action_idx = self.select_action(state, self.env.board)
                next_state, reward, done, _ = self.env.step(action_idx)
                total_reward += reward

                self.memory.push((state, action_idx, reward, next_state, done))
                state = next_state

                self.steps_done += 1
                if self.epsilon > epsilon_end:
                    self.epsilon -= (epsilon_start - epsilon_end) / epsilon_decay

                if len(self.memory) > batch_size:
                    experiences = self.memory.sample(batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)

                    batch_state = torch.from_numpy(np.array(batch_state)).to(self.device)
                    batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
                    batch_reward = torch.FloatTensor(batch_reward).to(self.device)
                    batch_next_state = torch.from_numpy(np.array(batch_next_state)).to(self.device)
                    batch_done = torch.FloatTensor(batch_done).to(self.device)

                    q_values = self.policy_net(batch_state).gather(1, batch_action)
                    next_q_values = self.target_net(batch_next_state).max(1)[0].detach()
                    expected_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))

                    loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
                    self.losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Log loss to wandb
                    wandb.log({'loss': loss.item(), 'step': self.steps_done})

                if self.steps_done % target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            self.episode_rewards.append(total_reward)
            result = self.env.board.result()
            if result == '0-1':
                self.wins += 1
            elif result == '1/2-1/2':
                self.draws += 1
            else:
                self.losses_count += 1

            # Log metrics at each episode
            wandb.log({
                'episode': episode + 1,
                'epsilon': self.epsilon,
                'total_reward': total_reward,
                'steps_done': self.steps_done,
            })

            if (episode + 1) % 100 == 0:
                win_rate = self.wins / 100
                self.win_rates.append(win_rate)
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode + 1}, Epsilon: {self.epsilon:.4f}, Win Rate: {win_rate:.2f}, Avg Loss: {avg_loss:.4f}")
                self.wins = 0
                self.draws = 0
                self.losses_count = 0

                # Save checkpoint
                checkpoint = {
                    'episode': episode,
                    'policy_net_state_dict': self.policy_net.state_dict(),
                    'target_net_state_dict': self.target_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'memory': self.memory,
                    'losses': self.losses,
                    'episode_rewards': self.episode_rewards,
                    'win_rates': self.win_rates,
                    'steps_done': self.steps_done
                }
                torch.save(checkpoint, self.model_path)
                print(f"Checkpoint saved at episode {episode + 1}")

                # Log metrics to wandb
                wandb.log({
                    'win_rate': win_rate,
                    'avg_loss': avg_loss,
                    'episode': episode + 1,
                })

    def save_final_model(self):
        torch.save({
            'episode': 1500 - 1,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory,
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
            'win_rates': self.win_rates,
            'steps_done': self.steps_done
        }, self.model_path)
        print("Final model saved.")

    def test_agent(self, num_games=10):
        agent_wins = 0
        opponent_wins = 0
        draws = 0

        for _ in range(num_games):
            state = self.env.reset()
            done = False

            while not done:
                if self.env.board.turn == chess.BLACK:
                    action_idx = self.select_action(state, self.env.board)
                    next_state, reward, done, _ = self.env.step(action_idx)
                    state = next_state
                else:
                    legal_moves = list(self.env.board.legal_moves)
                    move = random.choice(legal_moves)
                    self.env.board.push(move)
                    state = self.env.get_state()
                    done = self.env.board.is_game_over()

            result = self.env.board.result()
            if result == '0-1':
                agent_wins += 1
            elif result == '1/2-1/2':
                draws += 1
            else:
                opponent_wins += 1

        print(f"Agent Wins: {agent_wins}, Opponent Wins: {opponent_wins}, Draws: {draws}")

        # Log test results to wandb
        wandb.log({
            'test_agent_wins': agent_wins,
            'test_opponent_wins': opponent_wins,
            'test_draws': draws,
            'test_num_games': num_games,
        })

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='DQN Chess Agent Training')
    parser.add_argument('--model-path', type=str, default='chess_dqn_model.pth',
                        help='Path to save/load the model')
    args = parser.parse_args()
    model_path = args.model_path

    # Initialize components
    env = ChessEnv()
    agent = ChessAgent(ACTION_SIZE, model_path, env, device)

    # Initialize wandb
    wandb.init(project='dqn-chess-agent', name='training-run', config={
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.4,
        'epsilon_decay': 1000000,
        'learning_rate': 1e-4,
        'memory_capacity': 1000,
        'target_update': 1000,
        'num_episodes': 500,
    })
    wandb.watch(agent.policy_net, log='all')

    agent.load_model()
    agent.train(num_episodes=500, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.4, epsilon_decay=1000000, learning_rate=1e-4, memory_capacity=1000, target_update=1000)
    agent.save_final_model()
    agent.test_agent(num_games=100)

    # Finish wandb run
    wandb.finish()