import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Load Model dan Parameter
# ===============================
model_path = "dqn_pendulum_1000episode.h5"
model = keras.models.load_model(model_path, compile=False)  # ✅ tambahkan compile=False
print(f"✅ Model '{model_path}' berhasil dimuat tanpa kompilasi ulang.")

env = gym.make("Pendulum-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_space = [-2.0, 0.0, 2.0]
action_size = len(action_space)

# ===============================
# 2️⃣ Fungsi untuk memilih aksi
# ===============================
def select_action(state):
    q_values = model.predict(state, verbose=0)
    action_idx = np.argmax(q_values[0])
    return action_space[action_idx]

# ===============================
# 3️⃣ Pengujian Model
# ===============================
test_episodes = 10
reward_history = []

for episode in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for t in range(200):
        action = [select_action(state)]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = np.reshape(next_state, [1, state_size])

        if done:
            break

    reward_history.append(total_reward)
    print(f"🎯 Episode {episode+1}/{test_episodes} | Total Reward: {total_reward:.2f}")

env.close()

# ===============================
# 4️⃣ Visualisasi Hasil Pengujian
# ===============================
plt.plot(reward_history, marker='o')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Pendulum Testing Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_pendulum_testing_result.png")
plt.show()
