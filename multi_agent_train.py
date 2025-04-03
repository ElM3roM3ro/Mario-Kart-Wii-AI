import os
import time
import torch
import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
from collections import deque
from agent import BTRAgent  # Agent now uses the new PER with LazyFrames
from multiprocessing import shared_memory
from gym.wrappers import LazyFrames
import concurrent.futures

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
Ymem = 128
Xmem = 128  # Must include extra debug columns
data_shape = (Ymem + 1, Xmem)

# ----- Epsilon-Greedy Schedule Parameters -----
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_FRAMES = 2_000_000   # 2M frames
EPSILON_DISABLED_FRAMES = 100_000_000  # 100M frames

# ----- Helper Functions -----
def wait_for_shared_memory(shm_name, timeout=60):
    start_time = time.time()
    while True:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            logging.info(f"Shared memory '{shm_name}' created.")
            return shm
        except FileNotFoundError:
            if time.time() - start_time > timeout:
                logging.error(f"Timeout: Shared memory '{shm_name}' not created within {timeout} seconds.")
                raise TimeoutError(f"Shared memory '{shm_name}' not created within {timeout} seconds.")
            logging.info(f"Waiting for shared memory '{shm_name}' to be created...")
            time.sleep(1)

def read_shared_state(shm_array, timeout=12.0):
    t0 = shm_array[0, 0]
    start_time = time.time()
    while True:
        time.sleep(0.0167)
        t = shm_array[0, 0]
        if t != t0:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("No update in shared memory; process may be unresponsive.")
    state = shm_array[1:, :].copy().astype(np.uint8)
    reward = float(shm_array[0, 3])
    terminal = float(shm_array[0, 4])
    speed = float(shm_array[0, 5])
    lap_progress = float(shm_array[0, 6])
    return state, reward, terminal, speed, lap_progress

def preprocess_frame_stack(frame_stack):
    frames = []
    for frame in frame_stack:
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).float().to('cuda')
        frames.append(frame_tensor)
    stacked = torch.cat(frames, dim=0)  # Shape: (4, 128, 128)
    return stacked

def write_action(shm_array, action):
    shm_array[0, 2] = action

def launch_dolphin_for_worker(worker_id):
    shm_name = f"dolphin_shared_{worker_id}"
    os.environ["SHM_NAME"] = shm_name
    user = "Nolan"
    if user == "Nolan":
        user_dir = f"C:\\Users\\nolan\\DolphinUserDirs\\instance_{worker_id}"
    elif user == "Zach":
        user_dir = f"C:\\Users\\Zachary\\DolphinUserDirs\\instance_{worker_id}"
    elif user == "Victor":
        user_dir = f"C:\\Users\\victo\\DolphinUserDirs\\instance_{worker_id}"
    os.makedirs(user_dir, exist_ok=True)
    paths = {
        "dolphin_path": r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe",
        "script_path": r"F:\MKWii_Capstone_Project\UPDATED_MKWii_Capstone\Mario-Kart-Wii-AI\env_multi.py",
        "savestate_path": r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav",
        "game_path": r"E:\Games\Dolphin Games\MarioKart(Compress).iso",
    }

    user = "Nolan"
    if user == "Nolan":
        paths["dolphin_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\dolphin-x64-framedrawn-stable\Dolphin.exe"
        paths["script_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\env_multi.py"
        paths["savestate_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
        paths["game_path"] = r"C:\Users\nolan\source\repos\dolphin\Source\Core\DolphinQt\MarioKart(Compress).iso"
    elif user == "Zach":
        paths["dolphin_path"] = r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe"
        paths["script_path"] = r"F:\MKWii_Capstone_Project\UPDATED_MKWii_Capstone\Mario-Kart-Wii-AI\env_multi.py"
        paths["savestate_path"] = r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
        paths["game_path"] = r"E:\Games\Dolphin Games\MarioKart(Compress).iso"
    elif user == "Victor":
        paths["dolphin_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\Dolphin.exe"
        paths["script_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\env_multi.py"
        paths["savestate_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
        paths["game_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\MarioKart(Compress).iso"

    cmd = (
        f'"{paths["dolphin_path"]}" '
        f'-u "{user_dir}" '
        f'--no-python-subinterpreters '
        f'--script "{paths["script_path"]}" '
        f'--save_state="{paths["savestate_path"]}" '
        f'--exec="{paths["game_path"]}"'
    )
    proc = subprocess.Popen(cmd, shell=False)
    logging.info(f"Vectorized Env {worker_id}: Launched Dolphin with command: {cmd}")
    return shm_name, proc

total_frames = 0

class VecDolphinEnv:
    def __init__(self, num_envs, frame_skip=4):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.shm_arrays = []
        self.frame_buffers = []
        self.env_shms = []
        self.process_handles = []
        self.shm_names = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_envs)

        for i in range(num_envs):
            shm_name, proc = launch_dolphin_for_worker(i)
            self.shm_names.append(shm_name)
            self.process_handles.append(proc)

        for i in range(num_envs):
            shm = wait_for_shared_memory(self.shm_names[i])
            self.env_shms.append(shm)
            shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)
            self.shm_arrays.append(shm_array)
            initial_frame, _, _, _, _ = read_shared_state(shm_array)
            fb = deque(maxlen=4)
            for _ in range(4):
                fb.append(initial_frame)
            self.frame_buffers.append(fb)
    
    def restart_worker(self, i):
        logging.error(f"Worker {i} not responding. Restarting worker and Dolphin process.")
        try:
            self.process_handles[i].terminate()
            self.process_handles[i].wait(timeout=15)
        except Exception as e:
            logging.error(f"Error terminating worker {i}: {e}")
        shm_name, proc = launch_dolphin_for_worker(i)
        self.shm_names[i] = shm_name
        self.process_handles[i] = proc
        shm = wait_for_shared_memory(shm_name)
        self.env_shms[i] = shm
        shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)
        self.shm_arrays[i] = shm_array
        try:
            initial_frame, _, _, _, _ = read_shared_state(shm_array)
        except TimeoutError as e:
            logging.error(f"Worker {i} failed to initialize after restart: {e}")
            initial_frame = np.zeros((Ymem, Xmem), dtype=np.uint8)
        fb = deque(maxlen=4)
        for _ in range(4):
            fb.append(initial_frame)
        self.frame_buffers[i] = fb

    def step(self, actions):
        global total_frames
        next_obs = [None] * self.num_envs
        total_rewards = [0.0] * self.num_envs
        terminals = [0] * self.num_envs

        def worker_step(i, action):
            acc_reward = 0.0
            term_flag = 0
            last_state = None
            for _ in range(self.frame_skip):
                write_action(self.shm_arrays[i], action)
                try:
                    state, reward, terminal, speed, lap_progress = read_shared_state(self.shm_arrays[i], timeout=12.0)
                except TimeoutError as e:
                    logging.error(f"Worker {i} timeout: {e}")
                    self.restart_worker(i)
                    try:
                        state, reward, terminal, speed, lap_progress = read_shared_state(self.shm_arrays[i], timeout=12.0)
                    except TimeoutError:
                        state = np.zeros((Ymem, Xmem), dtype=np.uint8)
                        reward = 0.0
                        terminal = 0
                        break
                last_state = state
                self.frame_buffers[i].append(state)
                acc_reward += reward
                if terminal > 0:
                    term_flag = 1
                    break
            if term_flag:
                fb = deque(maxlen=4)
                for _ in range(4):
                    fb.append(last_state)
                self.frame_buffers[i] = fb
            obs_tensor = preprocess_frame_stack(list(self.frame_buffers[i]))
            return i, obs_tensor, acc_reward, term_flag

        futures = []
        for i in range(self.num_envs):
            futures.append(self.executor.submit(worker_step, i, actions[i]))
        for future in concurrent.futures.as_completed(futures):
            i, obs_tensor, acc_reward, term_flag = future.result()
            next_obs[i] = obs_tensor
            total_rewards[i] = acc_reward
            terminals[i] = term_flag

        total_frames += (self.frame_skip * self.num_envs)
        if total_frames % 4000 == 0:
            logging.info(f"Frames: {total_frames}")
        return next_obs, total_rewards, terminals

    def close(self):
        """Close and unlink all shared memory objects."""
        self.executor.shutdown(wait=True)
        for shm in self.env_shms:
            try:
                shm.close()
                shm.unlink()
                logging.info(f"Closed and unlinked shared memory: {shm.name}")
            except Exception as e:
                logging.error(f"Error closing shared memory {shm.name}: {e}")

# ----- Loss Logging & Checkpointing Utilities -----
checkpoint_path = "vectorized_checkpoint.pt"

def save_checkpoint(agent, update_count):
    checkpoint = {
        'online_state_dict': agent.online_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'update_count': update_count
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at update count {update_count} to {checkpoint_path}")

def load_checkpoint(agent):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.online_net.load_state_dict(checkpoint['online_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.update_count = checkpoint.get('update_count', 0)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")

def plot_metrics(loss_logs, episode_rewards):
    if loss_logs:
        updates, losses = zip(*loss_logs)
        plt.figure()
        plt.plot(updates, losses, label="Loss")
        plt.xlabel("Update Count")
        plt.ylabel("Loss")
        plt.title("Loss vs Update Count")
        plt.legend()
        plt.savefig("loss_graph.png")
        plt.close()
        logging.info("Loss graph saved as loss_graph.png")
    else:
        logging.info("No loss logs to plot.")
    if episode_rewards:
        plt.figure()
        plt.plot(range(len(episode_rewards)), episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Reward Over Time")
        plt.legend()
        plt.savefig("reward_graph.png")
        plt.close()
        logging.info("Reward graph saved as reward_graph.png")
    else:
        logging.info("No episode rewards to plot.")

def main():
    global total_frames
    num_envs = 8
    env = None
    try:
        env = VecDolphinEnv(num_envs, frame_skip=4)
        agent = BTRAgent(
            state_shape=(4, 128, 128),
            num_actions=8,
            num_quantiles=8,
            gamma=0.997,
            lr=1e-4,
            buffer_size=1048576,
            batch_size=256,
            target_update_interval=500,
            n_steps=3,
            munchausen_alpha=0.9,
            munchausen_tau=0.03,
            munchausen_clip=-1.0,
            device='cuda',
            parallel_envs=num_envs
        )
        load_checkpoint(agent)
        total_steps = 0
        update_frequency = num_envs
        loss_logs = []
        episode_rewards = []

        while True:
            current_states = [LazyFrames(list(env.frame_buffers[i])) for i in range(num_envs)]
            obs_list = []
            for i in range(num_envs):
                obs_tensor = preprocess_frame_stack(list(env.frame_buffers[i]))
                obs_list.append(obs_tensor.unsqueeze(0))
            batch_obs = torch.cat(obs_list, dim=0).to('cuda')

            if total_steps < EPSILON_DECAY_FRAMES:
                epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (total_steps / EPSILON_DECAY_FRAMES)
            elif total_steps < EPSILON_DISABLED_FRAMES:
                epsilon = EPSILON_END
            else:
                epsilon = 0.0

            with torch.no_grad():
                quantiles, _ = agent.online_net(batch_obs)
                q_mean = quantiles.mean(dim=1)
                logging.info(f"Q-values: mean={q_mean.mean().item():.4f}, max={q_mean.max().item():.4f}, min={q_mean.min().item():.4f}")
                greedy_actions = q_mean.argmax(dim=1).cpu().numpy()
            actions = []
            for ga in greedy_actions:
                if np.random.rand() < epsilon:
                    actions.append(np.random.randint(0, agent.num_actions))
                else:
                    actions.append(int(ga))

            next_obs_list, rewards, terminals = env.step(actions)
            total_steps += num_envs
            next_states = [LazyFrames(list(env.frame_buffers[i])) for i in range(num_envs)]

            for i in range(num_envs):
                transition = (
                    current_states[i],
                    actions[i],
                    rewards[i],
                    terminals[i],
                    next_states[i]
                )
                agent.buffer.put(*transition, j=i)

            avg_reward = sum(rewards)/num_envs
            if total_steps % update_frequency == 0:
                loss_val = agent.update(total_frames)
                max_reward = max(rewards)
                min_reward = min(rewards)
                if loss_val is not None:
                    loss_logs.append((agent.update_count, loss_val))
                    episode_rewards.append(avg_reward)
                    logging.info(f"Update {agent.update_count}: Loss = {loss_val:.4f}, Avg. Reward = {avg_reward}, Min. Reward: {min_reward}, Max. Reward: {max_reward}")

            if total_steps % 3125 == 0:
                logging.info(f"Saving checkpoint at step: {total_steps}")
                save_checkpoint(agent, agent.update_count)

            i = 0
            for term, rew in zip(terminals, rewards):
                if term > 0:
                    episode_rewards.append(rew)
                    #logging.info(f"Worker {i}: episode ended with reward: {rew:.3f}")
                    #save_checkpoint(agent, agent.update_count)
                i += 1
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Exiting...")
        try:
            result = subprocess.run(
                'taskkill /F /IM Dolphin.exe',
                check=True,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("Dolphin instances closed successfully:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error closing Dolphin instances:")
            print(e.stderr)
    finally:
        plot_metrics(loss_logs, episode_rewards)
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
