import os
import torch

# Ensure compatibility of multiprocessing in Python with OpenMP (used by PyTorch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from nes_py.wrappers import JoypadSpace
import datetime
from pathlib import Path
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import ResizeObservation, SkipFrame
from metrics import MetricLogger
from agent import Mario


def main():
    # Patch the JoypadSpace environment wrapper to allow environment resets with additional arguments
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    # Initialize the environment with a specific version and render mode
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True,
                                    render_mode='human')

    # Restrict the action space to simplify the control scheme
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # Apply a series of wrappers to modify the environment’s observations
    env = SkipFrame(env, skip=4)  # Skip frames to reduce the frequency of agent decisions
    env = ResizeObservation(env, shape=84)  # Resize observations to 84x84 pixels
    env = GrayScaleObservation(env, keep_dim=False)  # Convert observations to grayscale
    env = TransformObservation(env, f=lambda x: x / 255.)  # Normalize pixel values
    env = FrameStack(env, num_stack=4)  # Stack 4 consecutive frames together

    # Reset the environment to its initial state
    env.reset()

    # Setup directories for saving training checkpoints
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = None  # Optionally set a checkpoint path to load a pre-trained model

    # Load pretrained model if required by uncommenting below
    # checkpoint = Path('checkpoints/2024-04-06T18-01-02/mario_net_13.chkpt')

    # Initialize the Mario agent with configurations
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir,
                  checkpoint=checkpoint)
    logger = MetricLogger(save_dir)  # Initialize the metric logger

    episodes = 20000  # Set the number of episodes for the training

    for episode in range(episodes):
        print("Episode: ", episode)
        state = env.reset()
        done = True

        while True:
            if done:
                env.reset()
            env.render()  # Render the environment to a screen or frame buffer

            action = mario.act(state)  # Decide an action based on the current state
            next_state, reward, done, truncated, info = env.step(action)  # Execute the action

            mario.cache(state, next_state, action, reward, done)  # Cache the results for learning
            q, loss = mario.learn()  # Update the agent from cached experiences

            logger.log_step(reward, loss, q)  # Log step-related metrics

            state = next_state  # Transition to the next state

            if done or info['flag_get']:  # Check if the episode should terminate
                break

        logger.log_episode()  # Log episode-related metrics

        if episode % 10 == 0:
            logger.record(
                episode=episode,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )  # Record periodic metrics

    env.close()  # Close the environment


if __name__ == "__main__":
    main()
