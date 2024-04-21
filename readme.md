# Super Mario Bros Reinforcement Learning Agent

This project is a reinforcement learning framework developed to train an agent to play the 
Super Mario Bros game via the OpenAI Gym interface with the `gym-super-mario-bros` environment.

## Project Description

The purpose of this project is to implement and train a deep reinforcement learning agent that 
learns to navigate through the levels of the Super Mario Bros game. The agent uses a convolutional 
neural network to make decisions based on the current state of the game, represented by processed 
frames from the game environment.

## Features

- **Environment Wrapping**: Utilizes multiple wrappers to preprocess the game states into a simplified and manageable format for the agent.
- **Epsilon-Greedy Strategy**: Implements an epsilon-greedy strategy for action selection to balance exploration and exploitation.
- **Double Deep Q-Network**: Uses a MarioNet, a convolutional neural network, to estimate Q-values for action selection.
- **Experience Replay**: Employs a replay buffer to store and reuse past experiences to update the network weights.
- **Frame Skipping and Stacking**: Frame skipping for faster decision making and frame stacking to give the agent a sense of motion.
- **Checkpointing**: Supports saving and loading trained models to resume training or for evaluation purposes.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**
   - ```git clone https://github.com/raghul1998/fai_project.git```
2. **Install Dependencies**
   - Ensure that Python 3.8+ is installed on your machine.
   - To install PyTorch, get command from PyTorch website (https://pytorch.org/) according to your system requirements.
   - Install required Python packages:
     ```
     pip install -r requirements.txt
     ```
3. **Run the Script**
    - Run ```python main.py```
    - If you want to load a model/checkpoint, comment line 45 and uncomment line 48
    - If GPU is available, it will be used automatically.
    - If you get any memory issues, change the LazyMemmapStorage in agent.py file at line 24 to a smaller value.
4. **To evaluate a trained Model**
   - Run ```python replay.py```

## Configuration
The `agent.py` file contains several configuration settings that determine the training behavior, 
such as the number of episodes, epsilon values, learning rate, and more. Modify these settings as 
needed to tune the agent's performance.

## Acknowledgements
   - OpenAI Gym for the gaming environment.
   - The NES Py project and `gym-super-mario-bros` for providing the NES environment.


