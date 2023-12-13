"""Env wrappers"""
from gym import ActionWrapper, Wrapper, logger, RewardWrapper
from gym.spaces import Discrete, Box
import numpy as np


class WrapPendulum(ActionWrapper):
    """ Wrap pendulum. """

    @property
    def action_space(self):
        return Discrete(2)

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value

    def action(self, action):
        return 4 * np.array(action)[np.newaxis] - 2


class WrapContinuousPendulum(ActionWrapper):
    """ Wrap Continuous Pendulum. """

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,))

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value

    def action(self, action):
        return np.clip(2 * action, -2, 2)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class WrapContinuousPendulumSparse(RewardWrapper, ActionWrapper):
    """ Combined Wrapper for Continuous Pendulum with Modified Action and Sparse Reward. """
    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(1,))

    @action_space.setter
    def action_space(self, value):
        self.env.action_space = value

    def action(self, action):
        # Scale and clip the action
        return np.clip(2 * action, -2, 2)

    def reward(self, reward):
        # Get the environment's current state, e.g., angle and angular velocity
        th, thdot = self.state  # Assumed that self.env.state is accessible

        # Define the balanced condition with reasonable thresholds
        angle_threshold = 0.1  # ±0.1 radians
        velocity_threshold = 0.5  # ±1 radian/second

        if abs(angle_normalize(th)) < angle_threshold and abs(thdot) < velocity_threshold:
            # Sparse reward scaled by dt when the pendulum is near balanced
            return 100 * self.dt
        else:
            # No reward when the pendulum is not in the balanced state
            return 0


class TimeLimit(Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def _past_limit(self):
        """Return true if we are past our limit"""
        if self._max_episode_steps is not None and self._max_episode_steps <= self._elapsed_steps:
            logger.debug("Env has passed the step limit defined by TimeLimit.")
            return True

        return False

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if info is None:
            info = {}
        info["time_limit"] = False

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                self.reset()  # automatically reset the env
            info["time_limit"] = True
            done = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()
