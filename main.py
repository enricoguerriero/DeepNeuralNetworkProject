import yaml
import wandb
from architectures.alpha4 import load_or_create_model, create_action_mapping, pretrain_model, self_play
from architectures.alpha5 import DQNAgent
from architectures.alpha6 import ChessEnv, ChessAgent
from pettingzoo.classic import connect_four_v3
from pettingzoo.classic import chess_v6
from pettingzoo.classic import tictactoe_v3
import architectures.dqn_tictactoe as dqn_tictactoe
import architectures.dqn_connect4 as dqn_connect4
import architectures.dqn_chess as dqn_chess

if __name__ == '__main__':

    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    model = config['model']
    
    wandb.login() 
    wandb.init(project = 'chess-qnn',
        name = model['run_name'],
        config = config)
    
    if config['alpha1']['enabled']:
        alpha1 = config['alpha1']
        env = tictactoe_v3.env()
        model, rewards = dqn_tictactoe.train_dqn(env, num_episodes=alpha1['num_episodes'], gamma=alpha1['gamma'], learning_rate=alpha1['learning_rate'], 
                                                 memory_size=alpha1['memory_size'], save_interval=alpha1['save_interval'], batch_size = alpha1['batch_size'])
    
    if config['alpha2']['enabled']:
        alpha2 = config['alpha2']
        env = connect_four_v3.env()
        model, rewards = dqn_connect4.train_dqn(env, num_episodes=alpha2['num_episodes'], gamma=alpha2['gamma'], learning_rate=alpha2['learning_rate'],
                                                memory_size=alpha2['memory_size'], save_interval=alpha2['save_interval'], batch_size = alpha2['batch_size'])
    
    if config['alpha3']['enabled']:
        alpha3 = config['alpha3']
        env = chess_v6.env()
        model, rewards = dqn_chess.train_dqn(env, num_episodes=alpha3['num_episodes'], gamma=alpha3['gamma'], learning_rate=alpha3['learning_rate'],
                                             memory_size=alpha3['memory_size'], save_interval=alpha3['save_interval'], batch_size = alpha3['batch_size'])
    
    if config['alpha4']['enabled']:
        print("Alpha4 enabled")
        alpha4 = config['alpha4']
        model = load_or_create_model(model_path = model['path'])
        move_to_index, index_to_move = create_action_mapping()
        print("Model created")
        print("Pretraining ...")
        if alpha4['pretraining']['doit']:
            pretraining = alpha4['pretraining']
            pretrain_model(model['path'], num_samples=pretraining['num_samples'], max_depth=pretraining['max_depth'],
                       stockfish_depth=pretraining['stockfish_depth'], batch_size=pretraining['batch_size'],
                       epochs=pretraining['epochs'], validation_split=pretraining['validation_split'], learning_rate=pretraining['learning_rate'])
        print("Pretraining completed")
        print("Self-play training ...")
        if alpha4['training']['doit']:
            training = alpha4['training']
            memory = self_play(model= training['model'], num_games=training['num_games'], move_to_index=training['move_to_index'],
                               index_to_move=training['index_to_move'])
        print("Self-play training completed")
    
    
    if config['alpha5']['enabled']: 
        print("Alpha5 enabled")
        alpha5 = config['alpha5']
        agent = DQNAgent()
        print("Agent created")
        print("Model loading ...")
        if model['load']:
            agent.load_pretrained_weights("/models/" + model['name'])
        print("Model loaded")
        print("Supervised training ...")    
        if alpha5['pretraining']['doit']:
            pretraining = alpha5['pretraining']
            agent.supervised_train(dataset_path = pretraining['data_path'], 
                                epochs = pretraining['epochs'], 
                                batch_size = pretraining['batch_size'],
                                save_path = "/models/" + model['name'] + "_pretrained.h5")
        print("Supervised training completed")  
        print("Self-play training ...")
        if alpha5['training']['doit']:
            training = alpha5['training']
            agent.train(episodes = training['episodes'], 
                        batch_size = training['batch_size'],
                        save_path = "/models/" + model['name'] + ".h5")
        print("Self-play training completed")
        
    if config['alpha6']['enabled']:
        print("Alpha6 enabled")
        alpha6 = config['alpha6']
        env = ChessEnv()
        agent = ChessAgent(model_path="/models/" + model['name'])
        print("Agent created")
        print("Model loading ...")
        agent.load_model()
        print("Model loaded")
        print("Self-play training ...")
        agent.train(num_episodes=alpha6["episodes"], batch_size=alpha6["batch_size"], gamma=alpha6["gamma"], 
                    epsilon_start=alpha6["epsilon_start"], epsilon_end=alpha6["epsilon_end"], epsilon_decay=["epsilon_decay"],
                    learning_rate=alpha6["learning_rate"], memory_capacity=alpha6["memory_capacity"], target_update=alpha6["target_update"])
        print("Self-play training completed")
        agent.save_final_model()
        print("Final model saved")
        print("Testing agent ...")
        agent.test_agent(num_games=100)
        print("Testing completed")
        
    wandb.finish()

