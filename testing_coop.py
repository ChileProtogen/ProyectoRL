import os
import numpy as np
import torch
import supersuit as ss
import imageio

from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
from stable_baselines3 import PPO

# =============================
#   CONFIG
# =============================

MODEL_DIR = "./"   # Carpeta donde guardaste los 4 modelos
VIDEO_OUT = "./videos_coop"
os.makedirs(VIDEO_OUT, exist_ok=True)

device = torch.device("cpu")

# =============================
#   CARGA DE POLÍTICAS POR AGENTE
# =============================

def load_models(model_dir):
    models = {}

    # Caballeros
    models["knight_0"] = PPO.load(os.path.join(model_dir, "ppo_knight_0"), device=device)
    models["knight_1"] = PPO.load(os.path.join(model_dir, "ppo_knight_1"), device=device)

    # Arqueros
    models["archer_0"] = PPO.load(os.path.join(model_dir, "ppo_archer_0"), device=device)
    models["archer_1"] = PPO.load(os.path.join(model_dir, "ppo_archer_1"), device=device)

    print("Modelos cooperativos cargados correctamente.")
    return models


models = load_models(MODEL_DIR)



#  FUNCIÓN PARA GENERAR ACCIÓN

def get_action(agent_name, obs):
    """
    Retorna la acción usando el modelo apropiado según el agente.
    """
    if obs is None or np.all(obs == 0):
        return None  # muerto

    obs_flat = obs.flatten()   # <<< FIX CLAVE

    if "knight" in agent_name:
        idx = int(agent_name.split("_")[-1])
        model = models[f"knight_{idx}"]
        action, _ = model.predict(obs_flat, deterministic=True)
        return action

    elif "archer" in agent_name:
        idx = int(agent_name.split("_")[-1])
        model = models[f"archer_{idx}"]
        action, _ = model.predict(obs_flat, deterministic=True)
        return action

    return None


#   MÉTRICA DE COOPERACIÓN

def cooperation_score(obs_dict):
    knight_y = []
    archer_y = []

    for agent, obs in obs_dict.items():
        if obs is None:
            continue

        y = obs[0, 1]   # posición Y real

        if "knight" in agent:
            knight_y.append(y)
        elif "archer" in agent:
            archer_y.append(y)

    if not knight_y or not archer_y:
        return 0.0

    avg_knight_y = np.mean(knight_y)
    score = np.mean([(ay - avg_knight_y) for ay in archer_y])

    return float(score)




#   CREAR ENTORNO


env = knights_archers_zombies_v10.env(render_mode="rgb_array")
env = aec_to_parallel(env)
env = ss.black_death_v3(env)

print("Test cooperativo iniciado...")



#   TEST + VIDEO


NUM_EPISODES = 3
frames_per_episode = []
coop_scores = []

for ep in range(NUM_EPISODES):
    print(f"\n▶ Episodio {ep + 1}")

    obs, infos = env.reset()
    dones = {a: False for a in env.possible_agents}

    episode_frames = []
    local_coop_scores = []
    step = 0

    while not all(dones.values()):

        # Calcular score
        local_coop_scores.append(cooperation_score(obs))

        # Acciones por agente
        actions = {}
        for agent, ob in obs.items():
            if dones[agent]:
                continue

            act = get_action(agent, ob)
            if act is None:
                act = env.action_space(agent).sample()

            actions[agent] = act

        # step env
        obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {a: terminations[a] or truncations[a] for a in terminations}

        # Render
        frame = env.render()
        if frame is not None:
            episode_frames.append(frame)

        step += 1
        if step > 2500:
            print("Límite alcanzado.")
            break

    coop_scores.append(np.mean(local_coop_scores))
    frames_per_episode.append(episode_frames)

    print(f"CoopScore = {coop_scores[-1]:.3f}")


env.close()



#  Guardar videos
for i, frames in enumerate(frames_per_episode):
    fpath = os.path.join(VIDEO_OUT, f"kaz_coop_ep{i+1}.mp4")

    print(f"Guardando {fpath} ...")

    try:
        frames = [np.array(f, dtype=np.uint8) for f in frames]
        with imageio.get_writer(fpath, fps=20, codec="libx264", format="FFMPEG") as w:
            for f in frames:
                w.append_data(f)
        print("OK")
    except Exception as e:
        print("Error:", e)


print("\n==============================")
print("      RESULTADOS COOPERATIVOS")
print("==============================")
for i, s in enumerate(coop_scores):
    print(f"   Episodio {i+1}: CoopScore = {s:.3f}")
print("================================\n")
print("✔ Test completo. Videos creados.")
