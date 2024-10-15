# Deep Neural Network Project

by Marco and Enrico

### Deep Q-Learning

Project based on Deep Q-Learning architectures, the aim is to create a chess bot using Reinforcement Learning


### How to use it

To install all the necessary library, just run in the terminal "pip install -r requirements.txt".

7 different programs can be run:

- TicTacToe_qlearning.ipynb is a simple notebook that does "traditional" reinforcement learning on TicTacToe, the notebook can be run and in the last block you can play against the model you trained
- main.py: from that script you can run all the 6 scripts of DQL. They are named Alpha$N$, with $N$ going from 1 to 6. It is sufficient to run the file without modifying anything. In order to decide which model to run and with which parameters, you can open the file config.yaml, set the "enable" variable inside the model you want to run to True and then adjust parameters as you prefer
