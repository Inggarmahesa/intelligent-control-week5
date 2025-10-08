import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os
import csv
import matplotlib.pyplot as plt

# Buat folder "results" di direktori tempat script ini dijalankan
save_folder = os.path.join(os.getcwd(), "results")
os.makedirs(save_folder, exist_ok=True)
save_file = os.path.join(save_folder, "training_results.csv")

# Inisialisasi environment (CartPole dari Gymnasium)
env = gym.make("CartPole-v1")

# Parameter DRL
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Bangun model Deep Q-Network (DQN)
model = keras.Sequential([
    keras.layers.Input(shape=(state_size,)),   # Input layer biar clean
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# Fungsi pilih aksi (epsilon-greedy)
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # eksplorasi
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])  # eksploitasi

# Siapkan file CSV untuk simpan hasil
with open(save_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Score", "Epsilon"])  # header

# Proses training
episodes = 1000
scores = []  # simpan skor untuk plotting

for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0

    for time in range(500):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = np.reshape(next_state, [1, state_size])
        score += reward

        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print(f"Episode {episode+1}/{episodes} selesai, score: {score}, epsilon: {epsilon:.3f}")

            # Simpan hasil ke CSV
            with open(save_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([episode+1, score, round(epsilon, 3)])
            
            scores.append(score)
            break

    # Update model (experience replay)
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)

        for s, a, r, ns, d in minibatch:
            target = r
            if not d:
                target += gamma * np.amax(model.predict(ns, verbose=0)[0])

            target_f = model.predict(s, verbose=0)
            target_f[0][a] = target
            model.fit(s, target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training selesai!")
print(f"Hasil training tersimpan di: {save_file}")

# --- Plot Grafik Episode vs Score ---
plt.figure(figsize=(10,5))
plt.plot(scores, label="Score per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Score (Total Reward)")
plt.title("DQN Training Progress on CartPole")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_folder, "training_plot.png"))  # simpan grafik
plt.show()

print(f"Grafik tersimpan di: {os.path.join(save_folder, 'training_plot.png')}")
