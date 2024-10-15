import os
import chess
import chess.engine
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks, models
from collections import deque
from tqdm import tqdm
import wandb
from wandb.keras import WandbCallback

# Initialize Weights and Biases (wandb)
wandb.init(project="chess-rl-training", name="chess_rl_experiment")

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set TensorFlow to use GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus, 'GPU')
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)

# Define global variables
ACTION_SPACE_SIZE = 4672  # Placeholder; will be updated based on action mapping
MAX_GAME_LENGTH = 200     # Maximum number of moves per game
STOCKFISH_PATH = "./stockfish/stockfish-ubuntu-x86-64-avx2"
MODEL_PATH = './models/model_01_15.h5'

# Load the pre-trained model or create a new one
def load_or_create_model(model_path=MODEL_PATH):
    try:
        with tf.device('/GPU:0'):
            model = models.load_model(model_path)
        print("Pre-trained model loaded successfully.")
        model.summary()
    except (OSError, IOError):
        print("Saved model not found. Building a new model.")
        input_layer = layers.Input(shape=(14, 8, 8))
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        output_layer = layers.Dense(ACTION_SPACE_SIZE, activation='softmax')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy')
        model.summary()
    return model

# Create Action Mapping
def create_action_mapping():
    move_to_index = {}
    index_to_move = {}
    idx = 0
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            move = chess.Move(from_square, to_square)
            uci = move.uci()
            if uci not in move_to_index:
                move_to_index[uci] = idx
                index_to_move[idx] = move
                idx += 1
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                uci = move.uci()
                if uci not in move_to_index:
                    move_to_index[uci] = idx
                    index_to_move[idx] = move
                    idx += 1
    return move_to_index, index_to_move

# Split Board Representation
def split_dims(board):
    board3d = tf.zeros((14, 8, 8), dtype=tf.int8)
    for piece in chess.PIECE_TYPES:
        squares = board.pieces(piece, chess.WHITE)
        for square in squares:
            idx = divmod(square, 8)
            idx = (7 - idx[0], idx[1])
            indices = tf.constant([[piece - 1, idx[0], idx[1]]])
            updates = tf.constant([1], dtype=tf.int8)
            board3d = tf.tensor_scatter_nd_update(board3d, indices, updates)
        squares = board.pieces(piece, chess.BLACK)
        for square in squares:
            idx = divmod(square, 8)
            idx = (7 - idx[0], idx[1])
            indices = tf.constant([[piece + 5, idx[0], idx[1]]])
            updates = tf.constant([1], dtype=tf.int8)
            board3d = tf.tensor_scatter_nd_update(board3d, indices, updates)
    return board3d

# Mask Illegal Moves
def mask_illegal_moves(policy_output, board, move_to_index):
    legal_moves = list(board.legal_moves)
    mask = np.zeros(policy_output.shape)
    for move in legal_moves:
        if move.uci() in move_to_index:
            mask[move_to_index[move.uci()]] = 1
    masked_policy = policy_output * mask
    masked_policy_sum = np.sum(masked_policy)
    if masked_policy_sum > 0:
        masked_policy /= masked_policy_sum
    else:
        masked_policy = mask / np.sum(mask)
    return masked_policy

# Move Selection
def select_move(model, board, move_to_index, index_to_move):
    state = split_dims(board)
    state_for_prediction = tf.expand_dims(state, axis=0)
    with tf.device('/GPU:0'):
        policy_pred = model(state_for_prediction, training=False).numpy()
    policy_pred = policy_pred[0]
    masked_policy = mask_illegal_moves(policy_pred, board, move_to_index)
    move_index = np.random.choice(range(len(masked_policy)), p=masked_policy)
    move = index_to_move.get(move_index, None)
    if move is None or move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move

# Reward Function
def get_reward(board):
    result = board.result()
    if result == '1-0':
        return 1
    elif result == '0-1':
        return -1
    else:
        return 0

# Self-Play Mechanism
def self_play(model, num_games, move_to_index, index_to_move, max_game_length=MAX_GAME_LENGTH):
    memory = []
    for game_num in range(1, num_games + 1):
        board = chess.Board()
        game_memory = []
        while not board.is_game_over() and board.fullmove_number <= max_game_length:
            state = split_dims(board)
            policy_pred = model(tf.expand_dims(state, axis=0), training=False).numpy()[0]
            masked_policy = mask_illegal_moves(policy_pred, board, move_to_index)
            move_index = np.random.choice(range(len(masked_policy)), p=masked_policy)
            move = index_to_move.get(move_index, None)
            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))
            board.push(move)
            game_memory.append((state, policy_pred))
        reward = get_reward(board)
        for state, policy in game_memory:
            memory.append((state, policy, reward))
        print(f"Completed self-play game {game_num}/{num_games} with reward {reward}.")
    return memory

# Pretraining Data Generation
def random_board(max_depth=200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)
    for _ in range(depth):
        random_move = random.choice(list(board.legal_moves))
        board.push(random_move)
        if board.is_game_over():
            break
    return board

def stockfish(board, depth=0):
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score

def get_dataset(num_samples=100000, max_depth=200, stockfish_depth=1):
    boards = []
    values = []
    for _ in tqdm(range(num_samples)):
        board = random_board(max_depth=max_depth)
        value = stockfish(board, depth=stockfish_depth)
        if value is None:
            continue
        prob = 1 / (1 + 10 ** (-value / 400))
        boards.append(split_dims(board))
        values.append(prob)
    with tf.device('/GPU:0'):
        b = tf.stack(boards)
        v = tf.stack(values)
    return b, v

# Pretraining Function
def pretrain_model(model_path=MODEL_PATH, num_samples=100000, max_depth=200, stockfish_depth=1, batch_size=2048, epochs=100, validation_split=0.1, learning_rate=5e-4):
    x_train, y_train = get_dataset(num_samples=num_samples, max_depth=max_depth, stockfish_depth=stockfish_depth)
    print(x_train.shape, y_train.shape)

    try:
        with tf.device('/GPU:0'):
            model = models.load_model(model_path)
        print("Model loaded successfully.")
    except (OSError, IOError):
        print("Saved model not found. Building a new model.")
        with tf.device('/GPU:0'):
            model = models.Model(inputs=layers.Input(shape=(14, 8, 8)), outputs=layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
        model.summary()

    with tf.device('/GPU:0'):
        model.fit(x_train, y_train, 
                  batch_size=batch_size, 
                  verbose=1,
                  epochs=epochs, 
                  validation_split=validation_split,
                  callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                             callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4),
                             WandbCallback()])
    model.save(model_path)
    return model

# Main Function to Run Training
def main():
    global move_to_index, index_to_move, ACTION_SPACE_SIZE
    move_to_index, index_to_move = create_action_mapping()
    ACTION_SPACE_SIZE = len(move_to_index)
    print(f"Total action space size: {ACTION_SPACE_SIZE}")

    # Load or create model
    model = load_or_create_model()

    # Pretraining
    model = pretrain_model()

    # Self-play Training
    NUM_GAMES = 10
    memory = self_play(model, NUM_GAMES, move_to_index, index_to_move)
    print("Self-play completed.")

    # Training logic can be added here if needed, using the collected self-play data
    # Example:
    # x_train, y_train, rewards = zip(*memory)
    # Train the model further based on self-play data.

if __name__ == '__main__':
    main()