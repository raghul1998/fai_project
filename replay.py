import datetime
from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario  # Import custom Mario agent class
from metrics import MetricLogger  # Import metrics logging utility
from wrappers import ResizeObservation, SkipFrame  # Import custom wrappers

# Patch the JoypadSpace wrapper to reset environments with additional arguments.
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Create the Super Mario Bros environment with human-readable rendering.
env = gym_super_mario_bros.make('SuperMarioBros-v0',
                                apply_api_compatibility=True,
                                render_mode='human')

# Reduce the action space to simplify the game control.
env = JoypadSpace(
    env,
    [['right'],  # Move right
     ['right', 'A']]  # Move right and jump
)

# Apply custom wrappers to modify the game's observations and mechanics.
env = SkipFrame(env, skip=4)  # Skip 4 frames between agent's actions.
env = ResizeObservation(env, shape=84)  # Resize observations to 84x84.
env = GrayScaleObservation(env, keep_dim=False)  # Convert observations to grayscale.
env = TransformObservation(env, f=lambda x: x / 255.)  # Normalize pixel values.
env = FrameStack(env, num_stack=4)  # Stack 4 frames together to capture motion.

# Reset environment to start state.
env.reset()

# Setup directories for saving checkpoints.
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)  # Ensure the directory exists.

# checkpoint = None  # For a new one
# Load an existing checkpoint if available.
checkpoint = Path('checkpoints/2024-04-06T18-01-02/mario_net_13.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir,
              checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min  # Set exploration rate to minimum.

logger = MetricLogger(save_dir)  # Initialize the metric logger.

episodes = 100  # Set the number of episodes for training.

for episode in range(episodes):
    print("Episode: ", episode)
    state = env.reset()  # Reset environment to start a new episode.
    while True:
        env.render()  # Render the game.
        action = mario.act(state)  # Decide an action based on the current state.
        next_state, reward, done, truncated, info = env.step(action)  # Perform the action.
        print(f"Action: {action}, Reward: {reward}")  # Log action and reward.
        mario.cache(state, next_state, action, reward, done)  # Cache the results.
        logger.log_step(reward, None, None)  # Log step details.
        state = next_state  # Update the state.
        if done or info['flag_get']:  # Check if the episode has ended.
            break

    logger.log_episode()  # Log episode summary.

    if episode % 20 == 0:
        # Record and save metrics every 20 episodes.
        logger.record(
            episode=episode,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
