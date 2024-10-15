import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import chess
from tqdm import tqdm
import wandb
import os
import chess.pgn
import time
from torch.utils import data

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 17, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.view(8 * 8 * 17,-1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = (17, 8, 8)  # 12 planes for piece types, 8x8 board
        self.move_indices, self.index_moves = self.create_action_mapping()
        self.action_size = len(self.move_indices)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95       # Discount factor
        self.epsilon = 1.0      # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.move_indices, self.index_moves = self.create_action_mapping()

    def create_action_mapping(self):
        """Creates mappings between moves and indices."""
        all_squares = [chess.SQUARE_NAMES[i] for i in range(64)]
        move_indices = {}
        index_moves = {}
        idx = 0
        for from_square in all_squares:
            for to_square in all_squares:
                # Skip moves where from_square == to_square
                if from_square == to_square:
                    continue
                
                move_str = from_square + to_square
                promotions = [None, 'q', 'r', 'b', 'n']
                for promotion in promotions:
                    if promotion:
                        move = move_str + promotion
                    else:
                        move = move_str
                    try:
                        move_obj = chess.Move.from_uci(move)
                        # Ensure that promotions are only allowed when appropriate
                        if move_obj.promotion and not promotion:
                            continue  # Skip invalid promotions
                        if move not in move_indices:
                            move_indices[move] = idx
                            index_moves[idx] = move
                            idx += 1
                    except chess.InvalidMoveError:
                        continue
        return move_indices, index_moves


    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        """Epsilon-greedy action selection with legal moves."""
        if np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        legal_act_values = act_values[0, legal_moves]
        best_legal_action_idx = torch.argmax(legal_act_values).item()
        return legal_moves[best_legal_action_idx]

    def replay(self, batch_size):
        """Trains the model using a batch of experiences."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.model.train()
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f = target_f.clone().detach()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            outputs = self.model(state)
            loss = F.mse_loss(outputs[0][action], torch.tensor(target, dtype=torch.float32).to(self.device))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def board_to_state(self, board):
        """Converts the board to a state representation."""
        state = np.zeros((17, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        for position in piece_map:
            piece = piece_map[position]
            plane = self.piece_to_plane(piece)
            row = position // 8
            col = position % 8
            state[plane][row][col] = 1.0
                # Add player to move
        state[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        # Add castling rights
        state[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        state[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        state[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        state[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        return state

    def piece_to_plane(self, piece):
        """Maps a piece to a plane index."""
        piece_type = piece.piece_type
        color = piece.color
        plane = (piece_type - 1) + (0 if color == chess.WHITE else 6)
        return plane

    def get_reward(self, board, done):
        """Computes the reward based on the board state."""
        if done:
            result = board.result()
            if result == '1-0':
                return 1  # Win
            elif result == '0-1':
                return -1  # Loss
            else:
                return 0  # Draw
        else:
            # Reward for intermediate state (e.g., based on material count or mobility)
            reward = 0
            reward += len(list(board.legal_moves)) / 100  # Encourage more mobility
            material_count = sum([piece.piece_type for piece in board.piece_map().values()])
            reward += material_count / 1000  # Encourage material advantage
            return reward

    def train(self, episodes=1000, batch_size=32, WandB=None, save_path = 'model.pth'):
        """Training loop for the agent."""
        start = time.time()
        for e in tqdm(range(episodes), desc="Training", unit="episode"):
            board = chess.Board()
            state = self.board_to_state(board)
            done = False
            while not done:
                legal_moves = [self.move_indices[move.uci()] for move in board.legal_moves if move.uci() in self.move_indices]
                if not legal_moves:
                    break  # No legal moves available
                action_idx = self.act(state, legal_moves)
                action_move = self.index_moves[action_idx]
                move = chess.Move.from_uci(action_move)
                if move not in board.legal_moves:
                    reward = -10
                    next_state = state
                    done = True
                else:
                    board.push(move)
                    done = board.is_game_over()
                    reward = self.get_reward(board, done)
                    next_state = self.board_to_state(board)
                self.remember(state, action_idx, reward, next_state, done)
                state = next_state
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
            result = board.result() if board.is_game_over() else "Game in progress"
            print(f"Episode {e+1}/{episodes} - Result: {result} - Epsilon: {self.epsilon}")
            end = time.time()
            episode_time = end - start
            if WandB is not None:
                wandb.log({
                    "episode":e+1,
                    "result":result,
                    "reward":reward,
                    "epsilon":self.epsilon,
                    "episode_time":episode_time
                })
            start = time.time()
            
            # Save model every 50 episodes
            if (e + 1) % 50 == 0:
                checkpoint_dir = "checkpoints"
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{e+1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at episode {e+1}")
        
        torch.save(self.model.state_dict(), save_path)
            
    
    def supervised_train(self, dataset_path, epochs=1, batch_size=64, save_path = 'pretrained_model.pth'):
        """Pre-train the model using supervised learning."""
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        dataset = ChessDataset(dataset_path, self.move_indices)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        for epoch in tqdm(range(epochs), desc="Supervised Training", unit="epoch"):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0
            start = time.time()
            for states, actions in data_loader:
                states, actions = states.to(self.device), actions.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(states)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted_actions = torch.max(outputs, 1)  # Get the predicted class
                correct_predictions += (predicted_actions == actions).sum().item()  # Count correct predictions
                total_samples += actions.size(0)
                
            avg_loss = total_loss / len(data_loader)
            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
            end = time.time()
            epoch_time = end - start
            wandb.log({
                "training/loss":avg_loss,
                "training/accuracy":accuracy,
                "training/epoch_time":epoch_time
            })
            start = time.time()
        # Save the pre-trained model
        torch.save(self.model.state_dict(), save_path)
        
    def load_pretrained_weights(self, model_path = "pretrained_model.pth"):
        """Load pretrained weights into the model."""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"Loaded pretrained model from {model_path}")
        else:
            print(f"No model found at {model_path}, starting from scratch.")
    

class ChessDataset(data.IterableDataset):
    def __init__(self, pgn_file_path, move_indices):
        self.pgn_file_path = pgn_file_path
        self.move_indices = move_indices

    def __iter__(self):
        return self.data_generator()

    def data_generator(self):
        with open(self.pgn_file_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    board = game.board()
                    for move in game.mainline_moves():
                        state = self.board_to_state(board)
                        move_idx = self.move_indices.get(move.uci())
                        if move_idx is not None:
                            state_tensor = torch.tensor(state, dtype=torch.float32)
                            action_tensor = torch.tensor(move_idx, dtype=torch.long)
                            yield state_tensor, action_tensor
                        board.push(move)
                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue  # Skip to the next game

    def board_to_state(self, board):
        """Converts the board to a state representation."""
        state = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = board.piece_map()
        for position in piece_map:
            piece = piece_map[position]
            plane = self.piece_to_plane(piece)
            row = position // 8
            col = position % 8
            state[plane][row][col] = 1.0
        return state

    def piece_to_plane(self, piece):
        """Maps a piece to a plane index."""
        piece_type = piece.piece_type
        color = piece.color
        plane = (piece_type - 1) + (0 if color == chess.WHITE else 6)
        return plane


if __name__ == "__main__":
    # Instantiate and train the agent
    agent = DQNAgent()
    # load the pre-trained model
    agent.load_pretrained_weights()
    # pretrain the model
    agent.supervised_train("./data/data_01_15.pgn", 1, 64)
    # train the agent
    agent.train(episodes=10, batch_size=16)