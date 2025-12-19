
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  

from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np


# CALLBACK para registrar recompensas y graficar progreso
class RewardTracker(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.mean_rewards = []
        self.current_rewards = None
        self.num_envs = 1

    def _on_training_start(self):
        if hasattr(self.training_env, "num_envs"):
            self.num_envs = self.training_env.num_envs
        else:
            self.num_envs = 1
        self.current_rewards = np.zeros(self.num_envs)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if rewards is not None:
            self.current_rewards += rewards
            for i, done in enumerate(dones):
                if done:
                    self.episode_rewards.append(self.current_rewards[i])
                    self.current_rewards[i] = 0.0

        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % 10 == 0:
            mean_r = np.mean(self.episode_rewards[-10:])
            self.mean_rewards.append(mean_r)

        return True

    def _on_training_end(self):
        os.makedirs(self.save_path, exist_ok=True)
        plt.figure()
        plt.plot(self.mean_rewards, color="royalblue")
        plt.xlabel("Episodios (x10)")
        plt.ylabel("Recompensa media")
        plt.title("Evolución de recompensa - KAZ v10 sin cooperación")
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, "training_progress_v10.png"))
        plt.close()
        print(f"Gráfico de entrenamiento guardado en: {self.save_path}")


# CONFIGURACIÓN DEL ENTORNO
env = knights_archers_zombies_v10.env(render_mode=None)
env = aec_to_parallel(env)
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

# CALLBACK DE MONITOREO
callback = RewardTracker(save_path="./ppo_kaz_no_coop_v10/")

# OPCIÓN PARA CONTINUAR ENTRENAMIENTO
MODEL_PATH = "ppo_kaz_no_coop.zip"

if os.path.exists(MODEL_PATH):
    print("Modelo existente encontrado. Cargando y continuando entrenamiento...\n")
    model = PPO.load(MODEL_PATH, env=env, device="cpu")
else:
    print("No se encontró modelo previo. Iniciando entrenamiento desde cero...\n")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=512,
        learning_rate=3e-4,
        n_steps=4096,
        ent_coef=0.01,
        tensorboard_log="./ppo_kaz_no_coop_tb/",
        device="cpu"
    )

# ENTRENAMIENTO
model.learn(total_timesteps=2_000_000, callback=callback)

# GUARDAR MODELO
model.save("ppo_kaz_no_coop")
print("Entrenamiento completado y modelo guardado correctamente como 'ppo_kaz_no_coop'.")
