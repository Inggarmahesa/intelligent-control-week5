import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ Environment & Parameter
# ===============================
env = gym.make("Pendulum-v1")
state_size = env.observation_space.shape[0]
action_space = [-2.0, 0.0, 2.0]  # Discrete approximation
action_size = len(action_space)

learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
episodes = 1000
memory = deque(maxlen=50000)

# ===============================
# 2️⃣ Model DQN & Target Network
# ===============================
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(state_size,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(action_size, activation="linear")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# ===============================
# 3️⃣ Fungsi aksi epsilon-greedy
# ===============================
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# ===============================
# 4️⃣ Replay Training Function
# ===============================
def replay():
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    states = np.vstack([m[0] for m in minibatch])
    actions = np.array([m[1] for m in minibatch])
    rewards = np.array([m[2] for m in minibatch])
    next_states = np.vstack([m[3] for m in minibatch])
    dones = np.array([m[4] for m in minibatch])

    # Target Q menggunakan target network
    target_q = model.predict(states, verbose=0)
    next_q = target_model.predict(next_states, verbose=0)
    
    for i in range(batch_size):
        if dones[i]:
            target_q[i, actions[i]] = rewards[i]
        else:
            target_q[i, actions[i]] = rewards[i] + gamma * np.max(next_q[i])

    model.fit(states, target_q, epochs=1, verbose=0)

# ===============================
# 5️⃣ Training Loop
# ===============================
reward_history = []
update_target_every = 10  # update target network setiap 10 episode

for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for t in range(200):
        action_idx = select_action(state, epsilon)
        action = [action_space[action_idx]]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        # Normalisasi reward
        reward = (reward + 16) / 16.0

        memory.append((state, action_idx, reward, next_state, done))
        state = next_state
        total_reward += reward

        replay()

        if done:
            break

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if (episode + 1) % update_target_every == 0:
        target_model.set_weights(model.get_weights())

    reward_history.append(total_reward)
    print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}", flush=True)

# ===============================
# 6️⃣ Simpan model
# ===============================
model.save("dqn_pendulum_1000episode.keras")
model.save("dqn_pendulum_1000episode.h5") 
print("\n✅ Model tersimpan sebagai 'dqn_pendulum.keras'")
print("\n✅ Model tersimpan sebagai 'dqn_pendulum.h5'")

# ===============================
# 7️⃣ Plot hasil training
# ===============================
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward (Normalized)")
plt.title("DQN Training on Pendulum-v1 (Discrete Actions)")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_pendulum_training_result.png")
plt.show()

env.close()