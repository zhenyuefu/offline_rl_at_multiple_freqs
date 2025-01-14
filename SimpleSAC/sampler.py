import os
import numpy as np
import torch

from .utils import vid_from_frames, plot_q_over_traj

class StepSampler(object):

    def __init__(self, env, max_traj_length=1000, action_scale=1.0):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()
        self.action_scale = action_scale

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                np.expand_dims(observation, 0), deterministic=deterministic
            )[0, :]
            action = action / self.action_scale
            next_observation, reward, done, _ = self.env.step(action)
            # reward = reward * (fs/10)
            observations.append(observation)
            actions.append(action*self.action_scale)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action*self.action_scale, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000, action_scale=1.0):
        self.max_traj_length = max_traj_length
        self._env = env
        self.action_scale = action_scale

    def sample(self, policy, n_trajs, dt_feat, dt, deterministic=False, replay_buffer=None, video=False, output_file='', qs=None):
        trajs = []
        for traj in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []
            successes = []
            if video and traj == 0:
                imgs = []

            observation = self.env.reset()

            # if you want to play back actions at different dt, uncomment
            # import pickle
            # old_actions = pickle.load(open('actions.pkl', 'rb'))
            for _ in range(self.max_traj_length):
                if dt_feat:
                    observation = np.hstack([
                        observation, [dt]]).astype(np.float32)
                action = policy(
                    np.expand_dims(observation, 0), deterministic=deterministic
                )[0, :]
                action = action/self.action_scale
                next_observation, reward, done, info = self.env.step(action)
                observations.append(observation)
                actions.append(action*self.action_scale)
                rewards.append(reward)
                dones.append(done)
                if 'score' in info:
                    successes.append(info['score'])
                elif 'success' in info:
                    successes.append(info['success'])
                else:
                    successes.append(0)
                next_observations.append(next_observation)
                if video and traj == 0:
                   # if 'rgb_array' in self.env.metadata['render.modes']:
                      #  if 'kitchen' in self.env.spec.id:
                       #     from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
                        #    imgs.append(KitchenTaskRelaxV1.render(self.env, 'rgb_array'))
                       # else: # pendulum
                     imgs.append(self.env.render(mode='rgb_array'))
                   # else: # for metaworld
                    #    imgs.append(self.env.render(offscreen=True))


                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action*self.action_scale, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(dict(
                observations=np.array(observations, dtype=np.float32),
                actions=np.array(actions, dtype=np.float32),
                rewards=np.array(rewards, dtype=np.float32),
                next_observations=np.array(next_observations, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                successes=np.array(successes, dtype=np.float32),
            ))
            if video and traj == 0:
                imgs = np.stack(imgs, axis=0)
                vid_from_frames(imgs, output_file)
                file_path_stem = os.path.splitext(output_file)[0]
                if qs:
                    q_estimates = []
                    for q in qs:
                        q_estimates.append(
                            q(torch.Tensor(np.array(observations)).cuda(),
                            torch.Tensor(np.array(actions)).cuda()).cpu().detach().numpy())
                    plot_q_over_traj(
                        q_estimates, rewards, imgs, f'{file_path_stem}_q.jpg')

        return trajs

    @property
    def env(self):
        return self._env
