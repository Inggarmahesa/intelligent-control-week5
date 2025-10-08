import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os
import csv
import matplotlib.pyplot as plt
import time

# === Buat folder hasil ===
save_folder = os.path.join(os.getcwd(), "results")
os.makedirs(save_folder, exist_ok=True)
save_file = os.path.join(save_folder, "training_results.csv")

# === Inisialisasi environment dengan visual render ===
env = gym.make("CartPole-v1", render_mode="human")

# === Parameter DQN ===
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# === Model jaringan saraf (Deep Q-Network) ===
model = keras.Sequential([
    keras.layers.Input(shape=(state_size,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# === Fungsi pilih aksi (epsilon-greedy) ===
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# === Siapkan file CSV ===
with open(save_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Score", "Epsilon"])

# === Proses Training ===
episodes = 10  # bisa dinaikkan ke 1000 nanti
scores = []

for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0

    for time_step in range(500):
        # === Render tiap langkah ===
        env.render()
        time.sleep(0.01)  # biar nggak terlalu cepat tampilnya

        # Pilih aksi
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])
        score += reward

        # Simpan ke memory
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print(f"Episode {episode+1}/{episodes} | Score: {score:.2f} | Epsilon: {epsilon:.3f}")
            # Simpan hasil ke CSV
            with open(save_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode+1, score, round(epsilon, 3)])
            break

    # === Training jaringan (Experience Replay) ===
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for s, a, r, ns, d in minibatch:
            target = r
            if not d:
                target += gamma * np.amax(model.predict(ns, verbose=0)[0])
            target_f = model.predict(s, verbose=0)
            target_f[0][a] = target
            model.fit(s, target_f, epochs=1, verbose=0)

    # Kurangi eksplorasi
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    scores.append(score)

# === Simpan model ===
model_path = os.path.join(save_folder, "dqn_cartpole_model.h5")
model.save(model_path)
print(f"\nModel tersimpan di: {model_path}")

# === Simpan grafik ===
plt.figure(figsize=(10,5))
plt.plot(scores, color="blue")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("DQN Training Progress (CartPole-v1)")
plt.grid(True)
plot_path = os.path.join(save_folder, "training_plot.png")
plt.savefig(plot_path)
plt.show()

print(f"Grafik tersimpan di: {plot_path}")
print(f"Hasil training tersimpan di: {save_file}")

env.close()
print("\nTraining selesai! Semua hasil tersimpan di folder 'results/'")
