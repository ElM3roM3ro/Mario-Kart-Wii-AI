# multi_agent_train.py

import os
import time
import subprocess
import numpy as np
import torch
from gym.vector import AsyncVectorEnv
from agent import RainbowDQN

# --- Minimal Gym Wrapper that attaches to shared memory ---
import gym
from gym import spaces
from PIL import Image
from multiprocessing import shared_memory

# Use the same shared memory parameters as in env_multi.py.
target_width = 128
target_height = 128
num_frames = 4
SHM_ROWS = 1 + num_frames * target_height  # 513
SHM_COLS = target_width                    # 128
DATA_SHAPE = (SHM_ROWS, SHM_COLS)

class DolphinWrapper(gym.Env):
    """
    Minimal Gym wrapper that attaches to an existing shared memory block.
    It reads the full stacked 4-frame observation from shared memory.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, shm_name, width=target_width, height=target_height):
        super(DolphinWrapper, self).__init__()
        self.shm_name = shm_name
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(num_frames, height, width), dtype=np.uint8)
        self.action_space = spaces.Discrete(14)
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        self.shm_array = np.ndarray(DATA_SHAPE, dtype=np.float32, buffer=self.shm.buf)

    def _read_obs(self):
        flat_data = self.shm_array[1:, :].copy().astype(np.uint8)
        obs = flat_data.reshape(num_frames, self.height, self.width)
        return obs

    def reset(self):
        t0 = self.shm_array[0, 0]
        while True:
            time.sleep(0.05)
            t = self.shm_array[0, 0]
            if t != t0:
                break
        return self._read_obs()

    def step(self, action):
        self.shm_array[0, 2] = action
        t0 = self.shm_array[0, 0]
        timeout = 0
        while True:
            time.sleep(0.01)
            t = self.shm_array[0, 0]
            if t != t0:
                break
            timeout += 1
            if timeout > 1000:
                break
        obs = self._read_obs()
        reward = float(self.shm_array[0, 3])
        done = bool(self.shm_array[0, 4] > 0)
        info = {'speed': float(self.shm_array[0, 5]),
                'lap_progress': float(self.shm_array[0, 6])}
        return obs, reward, done, info

    def render(self, mode='human'):
        obs = self._read_obs()
        pil_img = Image.fromarray(obs[0])
        pil_img.show()

    def close(self):
        self.shm.close()

# --- Function to launch Dolphin for a given worker ---
def launch_dolphin_for_worker(worker_id):
    shm_name = f"dolphin_shared_{worker_id}"
    os.environ["SHM_NAME"] = shm_name
    # Adjust these paths to your system.
    dolphin_path = r"C:\Path\To\Dolphin.exe"
    script_path = r"C:\Path\To\env_multi.py"
    savestate_path = r"C:\Path\To\your_savestate.sav"
    game_path = r"C:\Path\To\Game.iso"
    user_dir = os.path.join(os.getcwd(), f"DolphinUserDirs\\instance_{worker_id}")
    os.makedirs(user_dir, exist_ok=True)
    cmd = (
        f'"{dolphin_path}" '
        f'-u "{user_dir}" '
        f'--no-python-subinterpreters '
        f'--script "{script_path}" '
        f'--save_state="{savestate_path}" '
        f'--exec="{game_path}"'
    )
    subprocess.Popen(cmd, shell=True)
    print(f"Worker {worker_id}: Launched Dolphin with command: {cmd}")
    # Wait until shared memory is created.
    shm = None
    while shm is None:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            print(f"Worker {worker_id}: Waiting for shared memory '{shm_name}'...")
            time.sleep(1)
    return shm_name

def make_env(worker_id):
    """
    Returns a function that creates a DolphinWrapper instance for a given worker.
    """
    def _init():
        return DolphinWrapper(shm_name=f"dolphin_shared_{worker_id}")
    return _init

def main():
    num_envs = 4
    # Launch Dolphin instances.
    for i in range(num_envs):
        launch_dolphin_for_worker(i)
    # Wait for Dolphin instances to fully initialize.
    time.sleep(10)
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)

    agent = RainbowDQN(
        state_shape=(num_frames, target_height, target_width),
        num_actions=14,
        num_atoms=51,
        v_min=-1.0,
        v_max=1.0,
        gamma=0.99,
        lr=1e-4,
        buffer_size=100000,
        batch_size=32,
        target_update_interval=1000,
        n_steps=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    num_episodes = 1000
    max_steps_per_episode = 1000

    for episode in range(num_episodes):
        obs = vec_env.reset()  # Shape: (num_envs, num_frames, height, width)
        episode_rewards = np.zeros(num_envs)
        dones = [False] * num_envs

        for step in range(max_steps_per_episode):
            actions = []
            for i in range(num_envs):
                obs_tensor = torch.from_numpy(obs[i]).float()
                action = agent.select_action(obs_tensor)
                actions.append(action)
            next_obs, rewards, dones, infos = vec_env.step(actions)
            for i in range(num_envs):
                transition = (obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                agent.store_transition(transition)
                episode_rewards[i] += rewards[i]
            obs = next_obs
            agent.update()
            if all(dones):
                break
        print(f"Episode {episode+1} rewards per env: {episode_rewards}")
        if (episode + 1) % 100 == 0:
            checkpoint_path = f"checkpoint_episode_{episode+1}.pth"
            torch.save({
                'episode': episode+1,
                'frame_count': agent.frame_count,
                'state_dict': agent.online_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    vec_env.close()

if __name__ == '__main__':
    main()

