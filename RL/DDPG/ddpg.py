from actor_acritic import MLPActorCritic
from buffer import ReplayBuffer
from utils import count_vars


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy
import time
import gymnasium as gym

from tqdm import trange


class DDPG:
    def __init__(
        self,
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(),
        seed=42,
        steps_per_epoch=4000,
        epochs=100,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-4,
        q_lr=1e-3,
        batch_size=100,
        start_steps=10000,
        updated_after=1000,
        update_every=50,
        act_noise=0.1,
        num_test_episodes=10,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=1,
    ):
        self.env = env_fn()
        self.test_env = env_fn()

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping:
        self.act_limit = self.env.action_space.high[0]

        self.ac = actor_critic(
            self.env.observation_space, self.env.action_space, **ac_kwargs
        )

        # Target policy
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks w.r.t optimizers
        for p in self.ac_targ.parameters():
            p.requires_grad_(False)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size
        )

        self.gamma = gamma
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.start_steps = start_steps
        self.action_noise = act_noise
        self.batch_size = batch_size
        self.updated_after = updated_after
        self.updated_every = update_every

        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.ac.q.parameters(), lr=q_lr)

    @property
    def var_counts(self):
        return tuple(count_vars(m) for m in [self.ac.pi, self.ac.q])

    def compute_loss_q(self, data):
        o = data["obs"]
        a = data["act"]
        r = data["rew"]
        o2 = data["obs2"]
        d = data["done"]

        # get the q value
        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma(1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        loss_info = dict(QVals=q.detach.numpy())
        return loss_q, loss_info

    def compute_loss_pi(self, data):
        o = data["obs"]
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        #

        for p in self.ac.q.parameters():
            p.requires_grad_(False)

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.ac.q.parameters():
            p.requires_grad_(True)

        # Update target networks
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, _ = self.test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                o, r, term, trun, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1

    def __call__(self):
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, _ = self.env.reset()
        ep_ret = ep_len = 0

        for t in trange(total_steps):
            if t > self.start_steps:
                a = self.get_action(o, self.action_noise)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, term, trun, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            d = term or trun

            d = False if ep_len == self.max_ep_len else d

            self.replay_buffer.store(o, a, r, o2, d)
            o = o2  # update observation

            if d or (ep_len == self.max_ep_len):
                o, _ = self.env.reset()
                ep_ret = ep_len = 0

            # update
            if t >= self.updated_after and t % self.updated_every == 0:
                for _ in range(self.updated_after):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch
            self.test_agent()


if __name__ == "__main__":
    env_fn = lambda: gym.make("Pendulum-v1")

    ddpg = DDPG(
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[256] * 2),
        gamma=0.99,
        epochs=50,
    )
    ddpg()
