""" Vectorizing a list of environments (see openai baselines) """
from multiprocessing import Pipe, Process
from envs.utils import CloudpickleWrapper, VecEnv
from envs.utils import tile_images
import numpy as np
from envs.env import Env

def worker(remote, env_wrapper):
    """
    :args remote: children side of pipe
    :args env_wrapper: pickled version of the environment
    """
    env = env_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            o, r, d, i = env.step(data)
            if d:
                o = env.reset()
            remote.send((o, r, d, i))
        elif cmd == 'reset':
            o = env.reset()
            remote.send(o)
        elif cmd == 'render':
            remote.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'seed':
            remote.send(env.seed(data))
        else:
            raise NotImplementedError

class SingleVecEnv(Env):
    """
    Fall back to this class when only a single environment is given.

    :args envs: a list of a single environment.
    """
    def __init__(self, envs):
        assert len(envs) == 1
        self._env = envs[0]

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, rew, done, info = self._env.step(action[0])
        if done:
            obs = self.reset()[0]
        return obs[np.newaxis, ...], np.array([rew]), np.array([done]), \
            {k: np.array(i) for k, i in info.items()}

    def reset(self):
        obs = self._env.reset()
        return obs[np.newaxis, ...]

    def render(self, mode='human'):
        return self._env.render(mode)

    def seed(self, seed):
        return self._env.seed(seed)

    def close(self):
        return self._env.close()

class SubprocVecEnv(VecEnv):
    """
    Execute several environment parallely.

    :args envs: a list of SIMILAR environment to run parallely
    """
    def __init__(self, envs):
        if envs:
            self.waiting = False
            self.closed = False
            self.envs = envs
            nenvs = len(envs)
            self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
            self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env)))
                       for (work_remote, env) in zip(self.work_remotes, envs)]
            for p in self.ps:
                p.daemon = True # if main crashes, crash all
                p.start()
            for remote in self.work_remotes:
                remote.close() # work_remote are only used in child processes

            # get spaces
            self.remotes[0].send(('get_spaces', None))
            observation_space, action_space = self.remotes[0].recv()
            VecEnv.__init__(self, len(envs), observation_space, action_space)
            self.reward_range = envs[0].reward_range
            self.metadata = envs[0].metadata

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), \
            {k: np.stack([i[k] for i in infos]) for k in infos[0]}

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def seed(self, seeds):
        """ Seeding environment """
        for remote, s in zip(self.remotes, seeds):
            remote.send(('seed', s))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode='human'):
        self.remotes[0].send(('render', None))
        img = self.remotes[0].recv()
        if mode == 'rgb_array':
            return img
        elif mode != 'human':
            raise NotImplementedError

    def full_render(self, mode='human'):
        for remote in self.remotes:
            remote.send(('render', None))
        imgs = [remote.recv() for remote in self.remotes]
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError


Env.register(SubprocVecEnv)

def VEnv(envs):
    if len(envs) == 1:
        return SingleVecEnv(envs)
    else:
        return SubprocVecEnv(envs)

if __name__ == '__main__':
    from envs.pusher import DiscretePusherEnv
    nenvs = 64
    envs = [DiscretePusherEnv() for _ in range(nenvs)]
    vec_env = SubprocVecEnv(envs)

    obs = vec_env.reset()
    T = 200

    for i in range(T):
        a = [vec_env.action_space.sample() for _ in range(nenvs)]
        obs, rews, dones, _ = vec_env.step(a)
        vec_env.render()
