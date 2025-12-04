# sac_discrete.py
# Discrete SAC for AntiJammingEnv-v0 with SB3-like API and custom metrics logging.

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# ==========================
# Basic MLP + Networks
# ==========================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    """State -> Q(s, a) for all discrete actions."""
    def __init__(self, obs_dim, n_actions, hidden_sizes=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions, hidden_sizes)

    def forward(self, x: torch.Tensor):
        # x: [B, obs_dim] -> [B, n_actions]
        return self.mlp(x)


class PolicyNetwork(nn.Module):
    """State -> categorical policy over actions."""
    def __init__(self, obs_dim, n_actions, hidden_sizes=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim, n_actions, hidden_sizes)

    def forward(self, x: torch.Tensor):
        logits = self.mlp(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return probs, log_probs


# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, device):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.as_tensor(np.array(obs, dtype=np.float32), device=device)
        next_obs = torch.as_tensor(np.array(next_obs, dtype=np.float32), device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
        return obs, actions, rewards, next_obs, dones


# ==========================
# SACDiscrete class (SB3-like)
# ==========================
class SACDiscrete:
    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env = None,
        tensorboard_log: str | None = None,
        learning_rate_q: float = 3e-4,
        learning_rate_pi: float = 3e-4,
        buffer_size: int = 200_000,
        batch_size: int = 512,
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_starts: int = 10_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        alpha: float = 0.05,
        autotune: bool = False,
        target_entropy_scale: float = 1.0,
        max_grad_norm: float = 10.0,
        log_interval: int = 10_000,
        eval_interval: int = 50_000,
        eval_episodes: int = 5,
        metrics_log_every: int = 1000,
        seed: int | None = None,
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        SB3-style discrete SAC:

        env:        a single AntiJammingEnv-v0 (wrapped with RecordEpisodeStatistics, TimeLimit, etc.)
        eval_env:   optional separate env for deterministic evaluation
        """

        self.env = env
        self.eval_env = eval_env
        self.learning_rate_q = learning_rate_q
        self.learning_rate_pi = learning_rate_pi
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.alpha = alpha
        self.autotune = autotune
        self.target_entropy_scale = target_entropy_scale
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.metrics_log_every = metrics_log_every
        self.verbose = verbose

        # seeding
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            try:
                self.env.reset(seed=seed)
                if self.eval_env is not None:
                    self.eval_env.reset(seed=seed + 1)
            except TypeError:
                pass

        # device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # obs / action dims
        obs_space = self.env.observation_space
        assert isinstance(self.env.action_space, gym.spaces.Discrete), "SACDiscrete expects Discrete actions"
        self.obs_dim = int(np.prod(obs_space.shape))
        self.n_actions = self.env.action_space.n

        # networks
        self.q1 = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.q1_target = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_net = PolicyNetwork(self.obs_dim, self.n_actions).to(self.device)

        self.q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.q_optimizer = optim.Adam(self.q_params, lr=self.learning_rate_q)
        self.pi_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate_pi)

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # entropy tuning
        self.autotune = autotune
        if autotune:
            target = -self.target_entropy_scale * np.log(self.n_actions)
            self.target_entropy = torch.tensor(target, device=self.device, dtype=torch.float32)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate_pi)
            self.alpha = float(self.log_alpha.exp().item())
        else:
            self.alpha = alpha

        # tensorboard
        self.writer = SummaryWriter(tensorboard_log) if tensorboard_log is not None else None

        # internal state
        self.global_step = 0

    # ---------- utils ----------
    def _obs_to_tensor(self, obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return torch.as_tensor(obs, device=self.device)

    def _batch_obs_to_tensor(self, obs):
        return torch.as_tensor(np.array(obs, dtype=np.float32), device=self.device)

    def select_action(self, obs, deterministic: bool = False) -> int:
        """Public method, like model.predict (but simpler)."""
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            probs, log_probs = self.policy_net(obs_t)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
        return int(action.item())

    def _evaluate_policy(self, global_step: int):
        if self.eval_env is None:
            return
        returns = []
        for _ in range(self.eval_episodes):
            obs, info = self.eval_env.reset()
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            done = False
            ep_ret = 0.0
            while not done:
                action = self.select_action(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_ret += reward
                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            returns.append(ep_ret)
        mean_ret = float(np.mean(returns))
        if self.writer is not None:
            self.writer.add_scalar("eval/mean_return", mean_ret, global_step)
        if self.verbose:
            print(f"[eval] step={global_step}  mean_return={mean_ret:.3f}")

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ---------- training ----------
    def learn(self, total_timesteps: int):
        env = self.env
        obs, info = env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        episode_return = 0.0
        episode_length = 0
        episode = 0

        # custom metrics accumulators
        metrics_n_steps = 0
        metrics_n_switch = 0.0
        metrics_n_crc = 0.0
        metrics_sum_power = 0.0

        while self.global_step < total_timesteps:
            # act
            if self.global_step < self.learning_starts:
                action = env.action_space.sample()
            else:
                action = self.select_action(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_obs_flat = np.asarray(next_obs, dtype=np.float32).reshape(-1)

            # add to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs_flat, done)

            # accumulate custom metrics
            metrics_n_steps += 1
            metrics_n_switch += info.get("switched", 0.0)
            metrics_n_crc += info.get("crc_success", 0.0)
            metrics_sum_power += info.get("power_w", 0.0)

            episode_return += reward
            episode_length += 1
            self.global_step += 1
            obs = next_obs_flat

            # end of episode
            if done:
                if self.writer is not None:
                    self.writer.add_scalar("charts/episode_return", episode_return, self.global_step)
                    self.writer.add_scalar("charts/episode_length", episode_length, self.global_step)
                episode += 1
                obs, info = env.reset()
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                episode_return = 0.0
                episode_length = 0

            # training updates
            if self.global_step >= self.learning_starts and self.global_step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    if len(self.replay_buffer) < self.batch_size:
                        break

                    (
                        batch_obs,
                        batch_actions,
                        batch_rewards,
                        batch_next_obs,
                        batch_dones,
                    ) = self.replay_buffer.sample(self.batch_size, self.device)

                    # critic target
                    with torch.no_grad():
                        probs_next, log_probs_next = self.policy_net(batch_next_obs)
                        q1_next = self.q1(batch_next_obs)
                        q2_next = self.q2(batch_next_obs)
                        min_q_next = torch.min(q1_next, q2_next)
                        soft_q_next = (probs_next * (min_q_next - self.alpha * log_probs_next)).sum(dim=-1)
                        target_q = batch_rewards + (1.0 - batch_dones) * self.gamma * soft_q_next

                    # critic loss
                    q1_vals = self.q1(batch_obs)
                    q2_vals = self.q2(batch_obs)
                    q1_a = q1_vals.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                    q2_a = q2_vals.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                    q1_loss = F.mse_loss(q1_a, target_q)
                    q2_loss = F.mse_loss(q2_a, target_q)
                    q_loss = q1_loss + q2_loss

                    self.q_optimizer.zero_grad()
                    q_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q_params, self.max_grad_norm)
                    self.q_optimizer.step()

                    # actor loss
                    probs, log_probs = self.policy_net(batch_obs)
                    q1_pi = self.q1(batch_obs)
                    q2_pi = self.q2(batch_obs)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    policy_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=-1).mean()

                    self.pi_optimizer.zero_grad()
                    policy_loss.backward()
                    self.pi_optimizer.step()

                    # entropy autotune
                    if self.autotune:
                        alpha_loss = (
                            probs.detach()
                            * (-self.log_alpha.exp() * (log_probs + self.target_entropy).detach())
                        ).sum(dim=-1).mean()
                        self.alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer.step()
                        self.alpha = float(self.log_alpha.exp().item())

                    # target networks
                    self._soft_update(self.q1_target, self.q1)
                    self._soft_update(self.q2_target, self.q2)

                # log losses
                if self.writer is not None:
                    self.writer.add_scalar("loss/q1", q1_loss.item(), self.global_step)
                    self.writer.add_scalar("loss/q2", q2_loss.item(), self.global_step)
                    self.writer.add_scalar("loss/policy", policy_loss.item(), self.global_step)
                    if self.autotune:
                        self.writer.add_scalar("loss/alpha", self.alpha, self.global_step)
                        self.writer.add_scalar("loss/alpha_loss", alpha_loss.item(), self.global_step)

            # flush custom metrics
            if (
                self.writer is not None
                and metrics_n_steps > 0
                and self.global_step % self.metrics_log_every == 0
            ):
                self.writer.add_scalar(
                    "custom/switch_rate", metrics_n_switch / metrics_n_steps, self.global_step
                )
                self.writer.add_scalar(
                    "custom/crc_success", metrics_n_crc / metrics_n_steps, self.global_step
                )
                self.writer.add_scalar(
                    "custom/mean_power", metrics_sum_power / metrics_n_steps, self.global_step
                )
                metrics_n_steps = 0
                metrics_n_switch = 0.0
                metrics_n_crc = 0.0
                metrics_sum_power = 0.0

            # console logging
            if self.verbose and self.global_step % self.log_interval == 0:
                print(
                    f"step={self.global_step}  episodes={episode}  "
                    f"last_ep_return={episode_return:.3f}"
                )

            # evaluation
            if (
                self.eval_env is not None
                and self.global_step >= self.learning_starts
                and self.global_step % self.eval_interval == 0
            ):
                self._evaluate_policy(self.global_step)

        if self.writer is not None:
            self.writer.close()
        return self
