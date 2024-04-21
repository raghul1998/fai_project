import datetime
from pathlib import Path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
env = gym_super_mario_bros.make('SuperMarioBros-v0',
                                apply_api_compatibility=True,
                                render_mode='human')

env = JoypadSpace(
    env,
    [['right'],
     ['right', 'A']]
)

env = SkipFrame(env, skip=4)
env = ResizeObservation(env, shape=84)
env = GrayScaleObservation(env, keep_dim=False)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# checkpoint = None
checkpoint = Path('checkpoints/2024-04-06T18-01-02/mario_net_13.chkpt')
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir,
              checkpoint=checkpoint)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for episode in range(episodes):
    print("Episode: ", episode)
    state = env.reset()
    while True:
        env.render()
        action = mario.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        mario.cache(state, next_state, action, reward, done)
        logger.log_step(reward, None, None)
        state = next_state
        if done or info['flag_get']:
            break

    logger.log_episode()

    if episode % 20 == 0:
        logger.record(
            episode=episode,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )

