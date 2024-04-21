import os

import torch

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
    # Setup the environment
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = gym_super_mario_bros.make('SuperMarioBros-v0',
                                    apply_api_compatibility=True,
                                    render_mode='human')
    # Let's only consider right and jump right action
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    # Preprocess
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=84)
    env = GrayScaleObservation(env, keep_dim=False)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    # reset
    env.reset()

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    # checkpoint = None
    checkpoint = Path('checkpoints/2024-04-06T18-01-02/mario_net_13.chkpt')

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir,
                  checkpoint=checkpoint)
    logger = MetricLogger(save_dir)

    episodes = 20000

    for episode in range(episodes):
        print("Episode: ", episode)
        state = env.reset()
        done = True
        # Play the game
        while True:
            if done:
                env.reset()
            # Visualize
            env.render()
            # Run the agent for the given state
            action = mario.act(state)
            # Perform action
            next_state, reward, done, truncated, info = env.step(action)
            if done:
                env.reset()
            # Store the data
            mario.cache(state, next_state, action, reward, done)
            # Learn from the data
            q, loss = mario.learn()
            # Log the data
            logger.log_step(reward, loss, q)
            # Update the states
            state = next_state
            # Check for termination
            if done or info['flag_get']:
                break

        logger.log_episode()

        if episode % 10 == 0:
            logger.record(
                episode=episode,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )

    env.close()


if __name__ == "__main__":
    main()
