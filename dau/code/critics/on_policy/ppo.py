import torch
from torch import Tensor
from critics.on_policy.online_critic import OnlineCritic
from gym.spaces import Box
from gym import Space
from models import MLP, NormalizedMLP
from abstract import ParametricFunction

class PPOCritic(OnlineCritic):
    """Implements PPO value estimation method.
    See
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    for details.

    :args gamma: physical discount factor
    :args dt: framerate
    :args v_function: value function parametric approximator
    :args eps_clamp: PPO epsilon clamping
    """
    def __init__(self, gamma: float, dt: float,
                 v_function: ParametricFunction, eps_clamp: float):
        OnlineCritic.__init__(self, gamma=gamma, dt=dt,
                              v_function=v_function)
        self._eps_clamp = eps_clamp

    def loss(self, v: Tensor, v_target: Tensor, old_v: Tensor) -> Tensor:
        assert old_v.shape == v.shape and v_target.shape == v.shape
        loss_unclipped = ((v - v_target.detach()) ** 2)
        v_clipped = old_v + torch.clamp(v-old_v, -self._eps_clamp, self.eps_clamp)
        loss_clipped = ((v_clipped - v_target.detach()) ** 2)
        return .5 * torch.max(loss_clipped, loss_unclipped).mean()

    @staticmethod
    def configure(dt: float, gamma: float, observation_space: Space,
                  nb_layers: int, hidden_size: int,
                  noscale: bool, eps_clamp: float, normalize: bool) -> "OnlineCritic":

        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        v_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1,
                         nb_layers=nb_layers, hidden_size=hidden_size)

        if normalize:
            v_function = NormalizedMLP(v_function)

        return PPOCritic(gamma, dt, v_function, eps_clamp)
