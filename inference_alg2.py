import torch
import gymnasium as gym
import numpy as np
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt


# models
class IDM(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, o_t, o_tp1):
        x = torch.cat([o_t, o_tp1], dim=1)
        return self.net(x)


class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, z_t):
        return self.net(z_t)


# Gym Wrapper: latent action → real action
class LatentPolicyEnvWrapper(gym.Wrapper):
    def __init__(self, env, decoder):
        super().__init__(env)
        self.decoder = decoder
        self.latent_dim = decoder.net[0].in_features
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.latent_dim,),
            dtype=np.float32
        )

    def step(self, latent_action):
        latent_tensor = torch.tensor(latent_action, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.decoder(latent_tensor).cpu().numpy()[0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


# Callback for logging rewards
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("rewards") is not None:
            reward = self.locals["rewards"][0]
            self.current_rewards.append(reward)

        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self.episode_rewards.append(sum(self.current_rewards))
            self.current_rewards = []

        return True


# main
if __name__ == "__main__":
    latent_dim = 32
    action_dim = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # decoder
    decoder = ActionDecoder(latent_dim, action_dim).to(device)
    decoder.load_state_dict(torch.load(
        "step_10000_k5/trained_models/all_tasks_decoder.pt", map_location=device))
    decoder.eval()

    # 環境 MuJoCo
    env = gym.make("Walker2d-v4", render_mode=None)
    wrapped_env = LatentPolicyEnvWrapper(env, decoder)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    # PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
        device=device
        )


    # log
    callback = RewardCallback()
    model.learn(total_timesteps=10000, callback=callback)

    # save model
    model.save("step_10000_k5/latent_policy_ppo")
    print("Latent Policy Training done")

    # reward plot
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Latent Policy Training Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("step_10000_k5/latent_policy_rewards.png")
    plt.show()
