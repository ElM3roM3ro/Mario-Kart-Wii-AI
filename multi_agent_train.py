import os
import time
import torch
import numpy as np
import subprocess
import multiprocessing as mp
import logging
import matplotlib.pyplot as plt
from collections import deque
from agent import BTRAgent  # Import the new BTR agent

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
Ymem = 78
Xmem = 94  # Must include extra debug columns
data_shape = (Ymem + 1, Xmem)

# ----- Helper Functions -----
from multiprocessing import shared_memory

def wait_for_shared_memory(shm_name, timeout=60):
    """Wait until the shared memory is created by env_multi.py, up to timeout seconds."""
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

def read_shared_state(shm_array):
    """
    Reads from shared memory:
      Row 0: metadata [timestep, timestep, action, reward, terminal, speed, lap_progress]
      Rows 1: current frame (grayscale image).
    Returns:
      state (np.array), reward, terminal, speed, lap_progress.
    """
    t0 = shm_array[0, 0]
    while True:
        time.sleep(0.05)
        t = shm_array[0, 0]
        if t != t0:
            break
    state = shm_array[1:, :].copy().astype(np.uint8)
    reward = float(shm_array[0, 3])
    terminal = float(shm_array[0, 4])
    speed = float(shm_array[0, 5])
    lap_progress = float(shm_array[0, 6])
    return state, reward, terminal, speed, lap_progress

def preprocess_frame_stack(frame_stack):
    """
    Converts a list of numpy arrays (each a single game frame) into a PyTorch tensor 
    of shape (4, 128, 128) using GPU-based resizing.
    """
    frames = []
    for frame in frame_stack:
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).float().to('cuda')
        frame_resized = torch.nn.functional.interpolate(frame_tensor.unsqueeze(0),
                                                        size=(128, 128),
                                                        mode='bilinear',
                                                        align_corners=False).squeeze(0)
        frames.append(frame_resized)
    stacked = torch.cat(frames, dim=0)  # Shape: (4, 128, 128)
    return stacked

def write_action(shm_array, action):
    """Writes the selected action to shared memory (column 2 of row 0)."""
    shm_array[0, 2] = action

def save_stacked_tensor_as_png(worker_id, frame_count, tensor):
    """
    Saves the stacked tensor to a folder structured as:
      tensor_stacks/worker_{worker_id}/stack_{frame_count}/
    Each channel is saved as a separate PNG image.
    """
    base_folder = "tensor_stacks"
    worker_folder = os.path.join(base_folder, f"worker_{worker_id}")
    os.makedirs(worker_folder, exist_ok=True)
    stack_folder = os.path.join(worker_folder, f"stack_{frame_count}")
    os.makedirs(stack_folder, exist_ok=True)
    for idx in range(tensor.size(0)):
        img = tensor[idx].cpu().numpy()
        file_path = os.path.join(stack_folder, f"image_{idx}.png")
        plt.imsave(file_path, img, cmap='gray')
        logging.info(f"Saved image for worker {worker_id}, frame {frame_count}, channel {idx} at {file_path}")

# ----- Dolphin & Environment Launch Setup -----
paths = {
    "dolphin_path": r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe",
    "script_path": r"F:\MKWii_Capstone_Project\UPDATED_MKWii_Capstone\Mario-Kart-Wii-AI\env_multi.py",
    "savestate_path": r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav",
    "game_path": r"E:\Games\Dolphin Games\MarioKart(Compress).iso",
}

user = "Zach"
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

def launch_dolphin_for_worker(worker_id):
    shm_name = f"dolphin_shared_{worker_id}"
    os.environ["SHM_NAME"] = shm_name
    if user == "Nolan":
        user_dir = f"C:\\Users\\nolan\\DolphinUserDirs\\instance_{worker_id}"
    elif user == "Zach":
        user_dir = f"C:\\Users\\Zachary\\DolphinUserDirs\\instance_{worker_id}"
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

# ----- Global Prioritized Replay Buffer for Multi-Agent Training -----
class GlobalPrioritizedReplayBuffer:
    def __init__(self, capacity=6000000, alpha=0.2, manager=None):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = manager.list() if manager is not None else []
        self.priorities = manager.list() if manager is not None else []
        self.pos = 0
        self.lock = mp.Lock()
        logging.info("Global prioritized replay buffer initialized.")
    def store(self, transition, priority=1.0):
        with self.lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
                self.priorities.append(priority)
            else:
                self.buffer[self.pos] = transition
                self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size, beta=0.4):
        with self.lock:
            total = len(self.buffer)
            if total == 0:
                return [], [], None
            prios = np.array(self.priorities, dtype=np.float32) ** self.alpha
            probs = prios / prios.sum()
            indices = np.random.choice(total, batch_size, p=probs)
            samples = [self.buffer[i] for i in indices]
            total_prob = probs[indices]
            weights = (total * total_prob) ** (-beta)
            weights /= weights.max()
            return samples, indices, torch.FloatTensor(weights).unsqueeze(1)
    def update_priorities(self, indices, new_prios):
        with self.lock:
            for idx, prio in zip(indices, new_prios):
                self.priorities[idx] = prio.item()

# ----- Worker Process -----
def worker_process(worker_id, global_buffer, global_weights, episode_rewards):
    logging.info(f"Worker {worker_id}: Starting.")
    shm_name = launch_dolphin_for_worker(worker_id)
    try:
        shm = wait_for_shared_memory(shm_name, timeout=60)
    except TimeoutError as e:
        logging.error(f"Worker {worker_id}: {e}. Exiting worker process.")
        return
    shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)
    # Create a local BTR agent with the new hyperparameters.
    agent = BTRAgent(
        state_shape=(4, 128, 128),
        num_actions=14,
        num_quantiles=32,
        gamma=0.997,
        lr=1e-4,
        buffer_size=1000000,
        batch_size=64,
        target_update_interval=500,
        n_steps=3,
        munchausen_alpha=0.9,
        munchausen_tau=0.03,
        munchausen_clip=-1.0,
        device='cuda'
    )
    if 'state_dict' in global_weights:
        agent.online_net.load_state_dict(global_weights['state_dict'])
        agent.online_net.to('cuda')
        logging.info(f"Worker {worker_id}: Loaded global weights.")
    local_frame_count = 0
    max_steps_per_episode = 1000
    frame_skip = 4  # Repeat the selected action for 4 frames.
    frame_buffer = deque(maxlen=4)
    # Initialize frame buffer with the first observed state.
    initial_frame, _, _, _, _ = read_shared_state(shm_array)
    for _ in range(4):
        frame_buffer.append(initial_frame)
    
    # (Optional) n-step buffer for multi-step returns.
    n_step_buffer = deque(maxlen=agent.n_steps)

    while True:
        episode_reward = 0.0
        steps = 0
        done = False
        try:
            while not done and steps < max_steps_per_episode:
                # Get the current observation (4-frame stack) from the frame buffer.
                obs_tensor = preprocess_frame_stack(list(frame_buffer))
                # Agent selects an action based on the current 4-frame observation.
                action = agent.select_action(obs_tensor)
                write_action(shm_array, action)
                
                total_reward = 0.0
                # Repeat the selected action for the next 'frame_skip' frames.
                for skip in range(frame_skip):
                    state_np, reward, terminal, speed, lap_progress = read_shared_state(shm_array)
                    frame_buffer.append(state_np)
                    total_reward += reward
                    steps += 1
                    local_frame_count += 1
                    episode_reward += reward
                    # If a terminal condition is encountered during any skip, break early.
                    if terminal > 0:
                        break

                # Build the next observation from the updated frame buffer.
                next_obs_tensor = preprocess_frame_stack(list(frame_buffer))
                # Create a transition with the accumulated reward and the new 4-frame stack.
                transition = (
                    obs_tensor.cpu().numpy().astype(np.uint8),  # current state (stack)
                    action,
                    total_reward,  # accumulated reward over the skipped frames
                    next_obs_tensor.cpu().numpy().astype(np.uint8),  # next state (stack)
                    terminal
                )
                agent.store_transition(transition)
                
                # Periodically update local network weights.
                if local_frame_count % 1000 == 0 and 'state_dict' in global_weights:
                    agent.online_net.load_state_dict(global_weights['state_dict'])
                    agent.online_net.to('cuda')
                    logging.info(f"Worker {worker_id}: Updated local weights at frame count {local_frame_count}")
                
                if terminal > 0:
                    logging.info(f"Worker {worker_id}: Episode ended after {steps} steps. Total Reward = {episode_reward:.3f}")
                    # Flush any remaining transitions in the n-step buffer if using multi-step returns.
                    while n_step_buffer:
                        n = len(n_step_buffer)
                        R = sum([agent.gamma ** i * n_step_buffer[i][2] for i in range(n)])
                        next_obs = n_step_buffer[-1][3]
                        done_flag = any(t[4] for t in n_step_buffer)
                        obs, action_val, _, _, _ = n_step_buffer[0]
                        global_buffer.store((obs, action_val, R, next_obs, done_flag), priority=1.0)
                        n_step_buffer.popleft()
                    # Reset the frame buffer using the last observed state.
                    frame_buffer.clear()
                    for _ in range(4):
                        frame_buffer.append(state_np)
                    done = True

            logging.info(f"Worker {worker_id}: Avg reward for this episode: {(episode_reward/steps):.3f}")
            episode_rewards.append(episode_reward)
        except Exception as e:
            logging.exception(f"Worker {worker_id}: Exception occurred: {e}")
            time.sleep(1)

# ----- Master Trainer Process -----
def master_trainer(global_buffer, global_weights, loss_logs, num_updates=1000000, batch_size=32, checkpoint_path='master_checkpoint.pt', checkpoint_interval=1000):
    logging.info("Master trainer: Initializing master network on CUDA.")
    master_agent = BTRAgent(
        state_shape=(4, 128, 128),
        num_actions=14,
        num_quantiles=32,
        gamma=0.997,
        lr=1e-4,
        buffer_size=1000000,
        batch_size=batch_size,
        target_update_interval=500,
        n_steps=3,
        munchausen_alpha=0.9,
        munchausen_tau=0.03,
        munchausen_clip=-1.0,
        device='cuda'
    )
    optimizer = master_agent.optimizer
    update_count = 0
    with torch.no_grad():
        cpu_state_dict = {k: v.cpu() for k, v in master_agent.online_net.state_dict().items()}
    global_weights['state_dict'] = cpu_state_dict
    while update_count < num_updates:
        samples, indices, weights = global_buffer.sample(batch_size, beta=min(1.0, 0.4 + update_count * (1.0-0.4)/100000))
        if len(samples) == 0:
            time.sleep(0.05)
            continue
        # Unpack batch and convert to tensors
        obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)
        obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0).float()/255.0 for o in obs_batch]).to('cuda')
        next_obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0).float()/255.0 for o in next_obs_batch]).to('cuda')
        actions = torch.LongTensor(actions).to('cuda')
        rewards = torch.FloatTensor(rewards).to('cuda')
        dones = torch.FloatTensor(dones).to('cuda')
        weights = weights.to('cuda')
        # --- Current state quantile estimates ---
        quantiles, taus = master_agent.online_net(obs_batch)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, master_agent.online_net.num_quantiles, 1)
        current_quantiles = quantiles.gather(2, actions_expanded).squeeze(2)
        # --- Munchausen bonus ---
        q_values = quantiles.mean(dim=1)
        logits = q_values / master_agent.munchausen_tau
        policy = torch.softmax(logits, dim=1)
        log_policy = torch.log_softmax(logits, dim=1)
        munchausen_bonus = torch.clamp(log_policy.gather(1, actions.unsqueeze(1)), min=master_agent.munchausen_clip)
        rewards_aug = rewards + master_agent.munchausen_alpha * munchausen_bonus.squeeze(1)
        # --- Next state quantile estimates ---
        next_quantiles_online, _ = master_agent.online_net(next_obs_batch)
        next_q_values_online = next_quantiles_online.mean(dim=1)
        best_actions = next_q_values_online.argmax(dim=1)
        best_actions_expanded = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, master_agent.online_net.num_quantiles, 1)
        next_quantiles_target, _ = master_agent.target_net(next_obs_batch)
        next_quantiles = next_quantiles_target.gather(2, best_actions_expanded).squeeze(2)
        # --- Compute target quantiles ---
        target_quantiles = rewards_aug.unsqueeze(1) + master_agent.gamma * next_quantiles * (1 - dones.unsqueeze(1))
        target_quantiles = target_quantiles.detach()
        # --- Quantile Huber loss ---
        taus = taus.unsqueeze(2)
        target_expanded = target_quantiles.unsqueeze(1)
        td_error = target_expanded - current_quantiles.unsqueeze(2)
        def huber_loss(x, k=1.0):
            cond = (x.abs() <= k).float()
            return 0.5 * x.pow(2) * cond + k * (x.abs() - 0.5 * k) * (1 - cond)
        loss_tensor = (torch.abs(taus - (td_error.detach() < 0).float()) * huber_loss(td_error, k=1.0)) 
        loss = loss_tensor.mean(dim=2).sum(dim=1)
        loss = (loss * weights.squeeze()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        new_priorities = loss.detach().abs().cpu() + 1e-6
        global_buffer.update_priorities(indices, new_priorities)
        update_count += 1
        if update_count % 100 == 0:
            loss_val = loss.item()
            loss_logs.append((update_count, loss_val))
            logging.info(f"Master update {update_count}: Loss = {loss_val:.4f}")
        if update_count % master_agent.target_update_interval == 0:
            master_agent.target_net.load_state_dict(master_agent.online_net.state_dict())
            logging.info("Master trainer: Updated target network.")
        with torch.no_grad():
            cpu_state_dict = {k: v.cpu() for k, v in master_agent.online_net.state_dict().items()}
        global_weights['state_dict'] = cpu_state_dict
        if update_count % checkpoint_interval == 0:
            torch.save({
                'online_state_dict': master_agent.online_net.state_dict(),
                'target_state_dict': master_agent.target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update_count': update_count
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at update {update_count} to {checkpoint_path}")
    logging.info("Master trainer: Training finished.")

# ----- Plotting Function -----
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

# ----- Main Multi-Agent Launch -----
def main():
    num_workers = 8  # Number of parallel Dolphin/agent instances.
    manager = mp.Manager()
    global_buffer = GlobalPrioritizedReplayBuffer(capacity=num_workers * 1000000, alpha=0.2, manager=manager)
    global_weights = manager.dict()
    loss_logs = manager.list()
    episode_rewards = manager.list()
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, global_buffer, global_weights, episode_rewards))
        p.start()
        workers.append(p)
        logging.info(f"Started worker process {i}.")
    try:
        master_trainer(global_buffer, global_weights, loss_logs, num_updates=1000000, batch_size=64)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    except Exception as e:
        logging.exception(f"Master trainer encountered an error: {e}")
    for p in workers:
        p.terminate()
        p.join()
        logging.info("Worker process terminated.")
    plot_metrics(list(loss_logs), list(episode_rewards))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
