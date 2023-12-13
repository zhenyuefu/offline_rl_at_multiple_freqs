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


class WrapContinuousPendulumSparse(RewardWrapper):
    def reward(self, reward):
        # 获取环境的当前状态，例如角度和角速度
        th, thdot = self.state

        # 定义平衡的条件
        if -0.25 <= th <= 0.25 and abs(thdot) < 0.5:
            return 10  # 摆锤处于平衡状态时的正奖励
        else:
            return 0  # 摆锤不处于平衡状态时的负奖励


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
