from actor_acritic import MLPActorCritic
from utils import count_vars
from buffer import VPGBuffer

import time

import torch
import torch.optim as optim


class VPG:
    def __init__(
        self,
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(),
        seed=42,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        logger_kwargs=dict(),
        save_freq=10,
    ):
        # self.env_fn = env_fn
        self.env = env_fn()

        # self.actor_critic = actor_critic
        # self.ac_kwargs = ac_kwargs
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        self.ac = actor_critic(
            self.env.observation_space, self.env.action_space, **ac_kwargs
        )
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.logger_kwargs = logger_kwargs
        self.save_freq = save_freq

        self.pi_optimizer = optim.Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)

    @property
    def var_counts(self):
        return tuple(count_vars(m) for m in [self.ac.pi, self.ac.v])

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy Losss
        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        data = self.buf.get()

        # Get Loss and info values
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        # Train policy with a single step of gradient descent
        self.pi_optimizer.zero_grad()

    def __call__(self):
        self.buf = VPGBuffer(
            self.obs_dim, self.act_dim, self.steps_per_epoch, self.gamma, self.lam
        )

        start_time = time.time()
        o, _ = self.env.reset()
        ep_ret = 0
        ep_len = 0

        for epoch in range(self.epochs):
            for t in range(self.steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, terminated, trunc, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                self.buf.store(o, a, r, v, logp)
                o = next_o

                timeout = ep_len == self.max_ep_len
                done = terminated or trunc or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if done or epoch_ended:
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0

                    self.buf.finish_path(v)
                    if done:
                        print("done!")
                    (
                        o,
                        _,
                    ) = self.env.reset()
                    ep_ret, ep_len = 0, 0
            self.update()


# Test
if __name__ == "__main__":
    import gymnasium as gym

    env_fn = lambda: gym.make("CartPole-v1")
    vpg = VPG(
        env_fn,
        MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[64] * 2),
        gamma=0.99,
        seed=42,
        steps_per_epoch=4000,
        epochs=50,
    )
    vpg()
