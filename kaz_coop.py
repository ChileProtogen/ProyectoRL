import os
import numpy as np
import gymnasium as gym

from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env



#   ROLE-SELECTOR ENV (single-agent Gym wrapper)

class SingleAgentKAZ(gym.Env):

    def __init__(self, role: str, controlled_idx: int):
        super().__init__()

        base = knights_archers_zombies_v10.env(render_mode=None)
        base = aec_to_parallel(base)

        self.env = base
        self.role = role

        agents = [a for a in self.env.possible_agents if a.startswith(role)]
        agents.sort()
        self.agent = agents[controlled_idx]

        # ACTION SPACE
        self.action_space = self.env.action_space(self.agent)

        # OBSERVATION SPACE FLATTEN
        orig_space = self.env.observation_space(self.agent)
        self.observation_space = gym.spaces.Box(
            low=orig_space.low.flatten(),
            high=orig_space.high.flatten(),
            shape=(orig_space.shape[0] * orig_space.shape[1],),
            dtype=orig_space.dtype
        )

        self.last_obs = None


    # ================= RESET =================
    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        self.last_obs = obs[self.agent].flatten()
        return self.last_obs, {}


    # ================= FORMATION REWARD =================
    def _formation_reward(self, obs_all):
        if not self.agent.startswith("knight"):
            return 0.0

        if obs_all[self.agent] is None:
            return 0.0

        # POSICIÓN Y DEL KNIGHT (ESCALAR)
        knight_y = float(obs_all[self.agent][0, 1])

        archer_ys = []
        for a in obs_all:
            if a.startswith("archer") and obs_all[a] is not None:
                archer_ys.append(float(obs_all[a][0, 1]))

        if not archer_ys:
            return 0.0

        avg_archer_y = float(np.mean(archer_ys))
        delta = knight_y - avg_archer_y  # >0 = delante

        # Reward shaping
        reward = np.clip(delta * 0.05, -1.0, 1.0)
        reward -= 0.01 * abs(delta)

        return float(reward)



    # ================= STEP =================
    def step(self, action):
        actions = {a: 0 for a in self.env.possible_agents}
        actions[self.agent] = action

        obs, rewards, terms, truncs, infos = self.env.step(actions)

        # --- FIX CLAVE: reward escalar ---
        base_reward = rewards[self.agent]
        if isinstance(base_reward, np.ndarray):
            base_reward = float(base_reward.item())
        else:
            base_reward = float(base_reward)

        shaping = float(self._formation_reward(obs))
        reward = base_reward + shaping

        terminated = terms[self.agent]
        truncated  = truncs[self.agent]

        self.last_obs = obs[self.agent].flatten()

        return self.last_obs, reward, terminated, truncated, infos[self.agent]


    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()



#   TRAIN FUNCTION 

def train_role(role, idx, timesteps, model_path):
    print(f"\n=== ENTRENANDO {role.upper()} #{idx} ===")

    env = SingleAgentKAZ(role=role, controlled_idx=idx)
    check_env(env, warn=True)

    if os.path.exists(model_path):
        print(f"Continuando desde {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print("Entrenando desde cero")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,
            verbose=1,
            device="cpu"
        )

    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    env.close()

    print(f"Guardado: {model_path}")



#   MAIN

if __name__ == "__main__":
    TIMESTEPS = 2_000_000  

    roles = [
        ("knight", 0, "ppo_knight_0.zip"),
        ("knight", 1, "ppo_knight_1.zip"),
        ("archer", 0, "ppo_archer_0.zip"),
        ("archer", 1, "ppo_archer_1.zip"),
    ]

    for role, idx, path in roles:
        train_role(role, idx, TIMESTEPS, path)
