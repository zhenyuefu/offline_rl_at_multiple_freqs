from typing import Union
from mylog import log
from logging import info
from memory.trajectory import Trajectory
from agents.on_policy.online_agent import OnlineAgent
from optimizer import setup_optimizer
from itertools import chain
from critics.on_policy.a2c import A2CCritic
from actors.on_policy.a2c import A2CActorContinuous, A2CActorDiscrete

TypeA2CActor = Union[A2CActorContinuous, A2CActorDiscrete]

class A2CAgent(OnlineAgent):
    """Synchronous Advantage Actor Critic Agent.

    :args T: number of max steps used for bootstrapping
       (to be computationnally efficient, bootstrapping horizon is variable).
    :args actor: actor used
    :args critic: critic used
    :args opt_name: 'rmsprop' ('sgd' deprecated)
    :args lr: unscaled learning rate
    :args dt: framerate
    :args weigth_decay: weight decay
    """
    def __init__(self, T: int, actor: TypeA2CActor, critic: A2CCritic,
                 opt_name: str, lr: float, dt: float, weight_decay: float):
        OnlineAgent.__init__(self, T=T, actor=actor, critic=critic)

        self._optimizer = setup_optimizer(
            chain(self._actor._policy_function.parameters(), self._critic._v_function.parameters()),
            opt_name=opt_name,
            lr=lr, dt=dt, inverse_gradient_magnitude=1, weight_decay=weight_decay)

    def learn(self) -> None:
        if self._count % self._T != self._T - 1:
            return None
        traj = Trajectory.tobatch(*self._current_trajectories)
        traj = traj.to(self._device)
        v, v_target = self._critic.value_batch(traj)

        critic_loss = self._critic.loss(v, v_target)
        critic_value = v_target - v

        obs = traj.obs
        actions = traj.actions
        distr = self._actor.actions_distr(obs)

        actor_loss = self._actor.loss(distr=distr, actions=actions,
                                      critic_value=critic_value)

        loss = critic_loss + actor_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        critic_loss = critic_loss.mean().item()
        critic_value = critic_value.mean().item()
        actor_loss = actor_loss.mean().item()

        info(f'At step {self._count}, critic loss: {critic_loss}')
        info(f'At step {self._count}, critic value: {critic_value}')
        info(f'At step {self._count}, actor loss: {actor_loss}')
        log("loss/critic", critic_loss, self._count)
        log("value/critic", critic_value, self._count)
        log("loss/actor", actor_loss, self._count)
        self._actor.log()
        self._critic.log()
