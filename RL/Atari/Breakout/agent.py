from replay_buffer import ReplayBuffer
from net import DQN
from transforms import Transforms


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

import time
from PIL import Image


class DQAgent:
    def __init__(
        self,
        replace_target_cnt,
        env,
        state_space,
        action_space,
        model_name="breakout_model",
        gamma=0.99,
        eps_start=0.1,
        eps_end=0.001,
        eps_dec=5e-6,
        batch_size=32,
        lr=0.001,
    ):
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.eps = eps_start
        self.eps_dec = eps_dec
        self.eps_end = eps_end

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.buffer = ReplayBuffer()

        # After how many training iterations the target network should updated
        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        # Initialize policy and target networks, set target network to eval mode
        self.policy_net = DQN(
            self.state_space, self.action_space, filename=model_name
        ).to(self.device)
        self.target_net = DQN(
            self.state_space, self.action_space, filename=model_name + "target"
        ).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        try:
            self.policy_net.load_model()
            print("Loaded pre-trained model")
        except:
            pass

        # Set target net to be the same as policy net
        self.replace_target_net()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = torch.nn.SmoothL1Loss()

    def sample_batch(self):
        batch = self.buffer.sample_batch(self.batch_size)
        state_shape = batch.state[0].shape

        # Convert to tensors with correct dimensions
        state = (
            torch.tensor(batch.state)
            .view(self.batch_size, -1, state_shape[1], state_shape[2])
            .float()
            .to(self.device)
        )
        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        state_ = (
            torch.tensor(batch.state_)
            .view(self.batch_size, -1, state_shape[1], state_shape[2])
            .float()
            .to(self.device)
        )
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    def greedy_action(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        action = self.policy_net(obs).argmax().item()

        return action

    def choose_action(self, obs):
        if random.random() > self.eps:
            action = self.greedy_action(obs)
        else:
            action = random.choice([x for x in range(self.action_space)])

        return action

    # Stores a transition into memory:
    def store_transition(self, *args):
        self.buffer.add_transition(*args)

    def replace_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Target network replaced")

    def dec_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_end else self.eps_end

    def learn(self, num_iters=1):
        # Samples a single batch according to batchsize and updates the policy net
        if self.buffer.pointer < self.batch_size:
            return

        for i in range(num_iters):
            # Sample batch
            state, action, reward, state_, done = self.sample_batch()

            # Calculate the value of the action taken
            q_eval = self.policy_net(state).gather(1, action)

            # Calculate best next action value from the target network
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze()

            # Using q_next and reward, calculate q_target
            # (1- done) ensures q_tart is 0 if transition is in a terminate state
            q_target = (1 - done) * (reward + self.gamma * q_next) + done * (reward)

            # Compute loss
            loss = self.loss(q_eval, q_target).to(self.device)

            # Perform backward propagation and optimization steps
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Increment learn_counter
            self.learn_counter += 1

            # Check replace target net
            self.replace_target_net()

        # Save model & decrement epsilon
        self.policy_net.save_model()
        self.dec_eps()

    def save_fit(self, num_transitions):
        frames = []
        for i in range(self.memory.pointer - num_transitions, self.memory.pointer):
            frame = Image.fromarray(self.memory.memory[i].raw_state, mode="RGB")
            frames.append(frame)

        frames[0].save(
            "episode.gif",
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=10,
            loop=0,
        )

    def train(self, num_eps=100, render=False):
        scores = []
        max_score = []

        for i in range(num_eps):
            done = False

            # Reset environment and pre-process state
            obs, _ = self.env.reset()
            state = Transforms.to_gray(obs)

            score = 0
            cnt = 0

            while not done:
                # Take epsilon greedy action
                action = self.choose_action(state)
                obs_, reward, terminated, truncated, _ = self.env.step(action)
                if render:
                    self.env.render()

                # Preprocess next state and store transtion
                state_ = Transforms.to_gray(obs, obs_)
                self.store_transition(state, action, reward, state_, int(done), obs)

                score += reward
                obs = obs_
                state = state_
                cnt += 1

            # Maintain record of the max score achieved so far
            if score > max_score:
                max_score = score

            # Save a gif if episode is best so far
            if score > 300 and score >= max_score:
                self.save_fit(cnt)

            scores.append(scores)
            print(
                f"Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
                \n\tEpsilon: {self.eps}\n\tTransitions added: {cnt}"
            )

            # Train on as many transition as there have been added in the episodes
            print(f"Learning x{math.ceil(cnt/self.batch_size)}")
            self.learn(math.ceil(cnt / self.batch_size))

        self.env.close()

    @torch.no_grad()
    def play_games(self, num_eps, render=True):

        # Set network to eval mode
        self.policy_net.eval()
        scores = []

        for i in range(num_eps):
            done = False

            # Get observation and pre-process
            obs = self.env.reset()
            state = Transforms.to_gray(obs)

            score = 0
            cnt = 0
            while not done:
                # Take the greedy action and observe next state

                action = self.greedy_action(state)
                obs_, reward, terminated, truncated, _ = self.env.step(action)

                if render:
                    self.env.render()

                # Pre-process next state and store transition
                state_ = Transforms.to_gray(obs, obs_)
                self.store_transition(state, action, reward, state_, int(done), obs)

                # Calculate score, set next state and obs and increment counter
                score += reward
                obs = obs_
                state = state_
                cnt += 1

            if score > 300:
                self.save_fit(cnt)

            scores.append(score)
            print(
                f"Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
                \n\tEpsilon: {self.eps}\n\tSteps made: {cnt}"
            )
