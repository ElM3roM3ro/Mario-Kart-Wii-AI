import os
import time
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import logging
from multiprocessing import shared_memory
from gym.vector import AsyncVectorEnv
import gym
from PIL import Image
from agent import RainbowDQN  # Your RainbowDQN implementation

# ----- Logging Setup -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)

# ----- Shared Memory Parameters -----
# These match the single-agent files.
Ymem = 78
Xmem = 94
data_shape = (Ymem + 1, Xmem)
# For vectorized environments, each worker uses its own shared memory name,
# e.g., "dolphin_shared_0", "dolphin_shared_1", etc.

# ----- DolphinWrapper (as defined in vec_env.py) -----
import collections

class DolphinWrapper(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, shm_name, frame_stack=4):
        super(DolphinWrapper, self).__init__()
        self.shm_name = shm_name
        self.frame_stack = frame_stack
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(frame_stack, Ymem, Xmem), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(14)
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        self.shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=self.shm.buf)
        self.frame_buffer = collections.deque(maxlen=frame_stack)

    def _read_single_frame(self):
        frame = self.shm_array[1:, :].copy().astype(np.uint8)
        return frame

    def _get_obs(self):
        return np.stack(list(self.frame_buffer), axis=0)

    def reset(self):
        t0 = self.shm_array[0, 0]
        while self.shm_array[0, 0] == t0:
            time.sleep(0.05)
        self.frame_buffer.clear()
        unique_frames = []
        last_frame = None
        while len(unique_frames) < self.frame_stack:
            current_frame = self._read_single_frame()
            if last_frame is None or not np.array_equal(current_frame, last_frame):
                unique_frames.append(current_frame)
                last_frame = current_frame
            time.sleep(0.05)
        self.frame_buffer = collections.deque(unique_frames, maxlen=self.frame_stack)
        return self._get_obs()

    def step(self, action):
        self.shm_array[0, 2] = action
        t0 = self.shm_array[0, 0]
        timeout = 0
        while True:
            time.sleep(0.01)
            if self.shm_array[0, 0] != t0:
                break
            timeout += 1
            if timeout > 1000:
                break
        new_frame = self._read_single_frame()
        self.frame_buffer.append(new_frame)
        obs = self._get_obs()
        reward = float(self.shm_array[0, 3])
        done = bool(self.shm_array[0, 4] > 0)
        info = {
            'speed': float(self.shm_array[0, 5]),
            'lap_progress': float(self.shm_array[0, 6])
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        obs = self._get_obs()
        pil_img = Image.fromarray(obs[0])
        pil_img.show()

    def close(self):
        self.shm.close()

# ----- Dolphin Launcher for Worker Environments -----
paths = {
    "dolphin_path": r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe",
    "script_path": r"F:\MKWii_Capstone_Project\UPDATED_MKWii_Capstone\Mario-Kart-Wii-AI\vec_env.py",
    "savestate_path": r"F:\MKWii_Capstone_Project\UPDATED_MKWii_Capstone\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav",
    "game_path": r"E:\Games\Dolphin Games\MarioKart(Compress).iso",
}
user = "Zach"
if user == "Nolan":
    paths["dolphin_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\dolphin-x64-framedrawn-stable\Dolphin.exe"
    paths["script_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\env_multi.py"
    paths["savestate_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
    paths["game_path"] = r"C:\Users\nolan\source\repos\dolphin\Source\Core\DolphinQt\MarioKart(Compress).iso"
elif user == "Victor":
    paths["dolphin_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\Dolphin.exe"
    paths["script_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\env_multi.py"
    paths["savestate_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
    paths["game_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\MarioKart(Compress).iso"

def launch_dolphin_for_worker(worker_id):
    shm_name = f"dolphin_shared_{worker_id}"
    os.environ["SHM_NAME"] = shm_name
    if user == "Zach":
        user_dir = f"C:\\Users\\Zachary\\DolphinUserDirs\\instance_{worker_id}"
    elif user == "Nolan":
        user_dir = f"C:\\Users\\nolan\\DolphinUserDirs\\instance_{worker_id}"
    elif user == "Victor":
        user_dir = f"C:\\Users\\victo\\DolphinUserDirs\\instance_{worker_id}"
    os.makedirs(user_dir, exist_ok=True)
    cmd = (
        f'"{paths["dolphin_path"]}" '
        f'-u "{user_dir}" '
        f'--no-python-subinterpreters '
        f'--script "{paths["script_path"]}" '
        f'--save_state="{paths["savestate_path"]}" '
        f'--exec="{paths["game_path"]}"'
    )
    subprocess.Popen(cmd, shell=True)
    logging.info(f"Worker {worker_id}: Launched Dolphin with command: {cmd}")
    return shm_name

# ----- Preprocessing Function -----
def preprocess_frame_stack(frame_stack):
    """
    Upscales each frame from (Ymem, Xmem) to (128, 128) using GPU bilinear interpolation.
    Returns a tensor of shape (4, 128, 128).
    """
    frames = []
    for frame in frame_stack:
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).float().to('cuda')
        frame_resized = F.interpolate(frame_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        frames.append(frame_resized)
    stacked = torch.cat(frames, dim=0)
    return stacked

# ----- Create Vectorized Environment -----
def make_env_fn(worker_id):
    def _init():
        return DolphinWrapper(shm_name=f"dolphin_shared_{worker_id}")
    return _init

def main():
    num_envs = 4  # For example, use 4 parallel Dolphin instances.
    # Launch Dolphin instances.
    for i in range(num_envs):
        launch_dolphin_for_worker(i)
    # Allow time for Dolphin to initialize and create shared memory.
    time.sleep(3)
    env_fns = [make_env_fn(i) for i in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)

    # Initialize RainbowDQN agent.
    agent = RainbowDQN(
        state_shape=(4, 128, 128),
        num_actions=14,
        num_atoms=51,
        v_min=-0.5,
        v_max=275.0,
        gamma=0.99,
        lr=1e-4,
        buffer_size=1000000,
        batch_size=32,
        target_update_interval=1000,
        n_steps=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    num_episodes = 1000000
    max_steps_per_episode = 1000

    # A simple local replay buffer (for demonstration purposes).
    replay_buffer = []

    def store_transition(transition):
        replay_buffer.append(transition)
        if len(replay_buffer) > agent.buffer.capacity:
            replay_buffer.pop(0)

    # Training loop.
    for episode in range(num_episodes):
        obs = vec_env.reset()  # obs shape: (num_envs, 4, Ymem, Xmem)
        episode_rewards = np.zeros(num_envs)
        dones = [False] * num_envs

        for step in range(max_steps_per_episode):
            actions = []
            for i in range(num_envs):
                # Preprocess: upscale each frame from (Ymem, Xmem) to (128, 128)
                obs_tensor = preprocess_frame_stack(obs[i])
                action = agent.select_action(obs_tensor)
                actions.append(action)
            next_obs, rewards, dones, infos = vec_env.step(actions)
            for i in range(num_envs):
                transition = (obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                store_transition(transition)
                agent.store_transition(transition)
                episode_rewards[i] += rewards[i]
            obs = next_obs
            agent.update()
            if all(dones):
                break

        logging.info(f"Episode {episode+1} rewards per env: {episode_rewards}")
        # Save checkpoint every 100 episodes.
        if (episode + 1) % 100 == 0:
            checkpoint_path = f"checkpoint_episode_{episode+1}.pth"
            torch.save({
                'episode': episode + 1,
                'frame_count': agent.frame_count,
                'state_dict': agent.online_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict()
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

    vec_env.close()

if __name__ == '__main__':
    main()
