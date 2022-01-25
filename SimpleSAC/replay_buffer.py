from copy import copy, deepcopy
import h5py
from queue import Queue
import threading

# import d4rl

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data['observations'].shape[0]:
                self._max_size = data['observations'].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._next_observations = np.zeros((self._max_size, observation_dim), dtype=np.float32)
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(next_observation, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[:self._size, ...],
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations[:self._size, ...],
            dones=self._dones[:self._size, ...]
        )


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }


# def get_d4rl_dataset(env):
#     dataset = d4rl.qlearning_dataset(env, dataset='halfcheetah-medium-v0')
#     return dict(
#         observations=dataset['observations'],
#         actions=dataset['actions'],
#         next_observations=dataset['next_observations'],
#         rewards=dataset['rewards'],
#         dones=dataset['terminals'].astype(np.float32),
#     )

def load_dataset(h5path):
    dataset_file = h5py.File(h5path, "r")
    dataset = dict(
        observations=dataset_file["obs"][:].astype(np.float32),
        actions=dataset_file["actions"][:].astype(np.float32),
        next_observations=dataset_file["next_obs"][:].astype(np.float32),
        rewards=dataset_file["rewards"][:].astype(np.float32),
        dones=dataset_file["dones"][:].astype(np.float32),
    )
    for k, v in dataset.items():
        if len(v.shape) > 1:
            dim_obs = v.shape[1]
        else:
            dim_obs = 1
        dataset[k] = v.reshape(-1, 256, dim_obs) # this only works for pendulum
    return dataset


def index_batch(batch, indices, size, n):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices[0], indices[1]].reshape(size, int(n), -1)
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size, n):
    ascending_idxs = np.tile(np.arange(n), size)
    traj_indices = np.random.randint(batch['observations'].shape[0]-n, size=size)
    traj_indices = np.repeat(traj_indices, n) + ascending_idxs
    batch_indices = np.random.randint(batch['observations'].shape[1], size=size)
    batch_indices = np.repeat(batch_indices, n)
    indices = np.vstack((traj_indices, batch_indices)).astype(int) # should be T, B
    return index_batch(batch, indices, size, n)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits
