import os
import numpy as np
import torch
import supersuit as ss
import imageio
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3 import PPO

# --- CONFIGURACIÓN ---
VIDEO_PATH = "./videos_no_coop"
os.makedirs(VIDEO_PATH, exist_ok=True)

# Forzar CPU 
device = torch.device("cpu")

# Cargar modelo entrenado 
model = PPO.load("ppo_kaz_no_coop", device=device)

# --- Crear entorno ---
env = knights_archers_zombies_v10.env(render_mode="rgb_array")  
env = aec_to_parallel(env)
env = ss.black_death_v3(env)

print("Iniciando test automático SIN cooperación (v10, grabando video)...")

# --- Ejecutar episodios ---
num_episodes = 3
frames_per_episode = []

for episode in range(num_episodes):
    print(f"\nEpisodio {episode + 1}")
    obs, infos = env.reset()
    dones = {agent: False for agent in env.possible_agents}
    episode_frames = []
    step = 0

    while not all(dones.values()):
        actions = {}
        for agent, ob in obs.items():
            if dones[agent]:
                continue
            if ob is None or np.all(ob == 0):
                actions[agent] = env.action_space(agent).sample()
            else:
                action, _ = model.predict(ob, deterministic=True)
                actions[agent] = action

        obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {a: terminations[a] or truncations[a] for a in terminations}

        frame = env.render()  # devuelve imagen RGB
        if frame is not None:
            episode_frames.append(frame)
        step += 1

        # Seguridad: evitar loops infinitos
        if step > 2500:
            print("Límite de pasos alcanzado.")
            break

    frames_per_episode.append(episode_frames)
    print(f"Episodio {episode + 1} completado ({len(episode_frames)} frames).")

env.close()

# --- Guardar los videos ---
for i, frames in enumerate(frames_per_episode):
    filename = os.path.join(VIDEO_PATH, f"kaz_no_coop_ep{i+1}.mp4")
    print(f"Guardando video {filename}...")
    try:
        # Convertir frames a uint8 y guardar usando ffmpeg
        frames = [np.array(frame, dtype=np.uint8) for frame in frames]
        with imageio.get_writer(filename, fps=20, codec="libx264", format="FFMPEG") as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Video guardado en {filename}")
    except Exception as e:
        print(f"Error al guardar {filename}: {e}")

print("\nTest completado y videos exportados correctamente.")
