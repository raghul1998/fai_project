import numpy as np
import gym
from gym.spaces import Box
from skimage import transform


class ResizeObservation(gym.ObservationWrapper):
    """
    A custom wrapper for OpenAI Gym environments that resizes observations to a specified shape.
    """

    def __init__(self, env, shape):
        """
        Initialize the ResizeObservation wrapper.

        Parameters:
        - env (gym.Env): The environment to wrap.
        - shape (int or tuple): The desired shape of the observation. If an integer is provided, it will create a square shape.
        """
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape,
                          shape)  # If shape is a single integer, convert it to a tuple for square resizing.
        else:
            self.shape = tuple(shape)  # Ensure the shape is a tuple.

        # Adjust the observation space dimensions according to the new shape.
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """
        Resize the observation from the environment.

        Parameters:
        - observation (np.array): The original observation from the environment.

        Returns:
        - np.array: The resized observation.
        """
        resize_obs = transform.resize(observation, self.shape,
                                      anti_aliasing=True)  # Resize observation
        resize_obs *= 255  # Scale pixel values back to [0, 255]
        resize_obs = resize_obs.astype(np.uint8)  # Convert floats to unsigned byte type
        return resize_obs


class SkipFrame(gym.Wrapper):
    """
    A custom wrapper for OpenAI Gym environments that skips a specified number of frames for each action.
    """

    def __init__(self, env, skip):
        """
        Initialize the SkipFrame wrapper.

        Parameters:
        - env (gym.Env): The environment to wrap.
        - skip (int): The number of frames to skip for each action.
        """
        super().__init__(env)
        self._skip = skip  # Number of frames to skip per action

    def step(self, action):
        """
        Execute an action and skip a defined number of frames, accumulating reward.

        Parameters:
        - action: The action to perform.

        Returns:
        - obs: The last observation after skipping frames.
        - total_reward: The total reward accumulated during frame skipping.
        - done: Boolean indicating if the episode is finished.
        - truncated: Boolean indicating if the episode was truncated.
        - info: A dictionary with extra information from the environment.
        """
        total_reward = 0.0
        done = False
        for i in range(self._skip):  # Loop over the number of frames to skip
            obs, reward, done, truncated, info = self.env.step(
                action)  # Take a step using the provided action
            total_reward += reward  # Accumulate rewards from each skipped frame
            if done:
                break  # Stop if the episode has ended
        return obs, total_reward, done, truncated, info  # Return the results from the last observation
