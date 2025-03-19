import os
import time
import torch
import numpy as np
import subprocess
import multiprocessing as mp
from collections import deque
from agent import RainbowDQN  # your agent implementation
import matplotlib.pyplot as plt

# ----- Shared Memory Parameters -----
Ymem = 78
Xmem = 94  # Must include extra debug columns (e.g., speed, lap_progress)
data_shape = (Ymem + 1, Xmem)

# ----- Helper Functions (as in single-agent train.py) -----
from multiprocessing import shared_memory

def wait_for_shared_memory(shm_name, timeout=60):
    """Wait until the shared memory is created by env.py, up to timeout seconds."""
    start_time = time.time()
    while True:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            print(f"Shared memory '{shm_name}' created.")
            return shm
        except FileNotFoundError:
            if time.time() - start_time > timeout:
                print(f"Timeout: Shared memory '{shm_name}' not created within {timeout} seconds.")
                raise TimeoutError(f"Shared memory '{shm_name}' not created within {timeout} seconds.")
            print(f"Waiting for shared memory '{shm_name}' to be created...")
            time.sleep(1)

def read_shared_state(shm_array):
    """
    Reads from shared memory:
      Row 0: [timestep, timestep, action, reward, terminal, speed, lap_progress]
      Rows 1: state as an image.
    Returns:
      state (np.array), reward, terminal, speed, lap_progress.
    """
    #print("Reading shared memory...")
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
    #print(f"Shared state read: Reward = {reward}, Terminal = {terminal}, Speed = {speed}, Lap Progress = {lap_progress}")
    return state, reward, terminal, speed, lap_progress

def preprocess_state(state):
    """
    Converts a state (np.array) to a PyTorch tensor of shape (4,128,128) using GPU-based resizing.
    """
    state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float().to('cuda')
    state_resized = torch.nn.functional.interpolate(state_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    stacked = state_resized.squeeze(0).repeat(4, 1, 1)
    return stacked

def write_action(shm_array, action):
    """Writes the selected action to shared memory (in column 2 of row 0)."""
    #print(f"Writing action: {action}")
    shm_array[0, 2] = action

# ----- New Helper: Save Stacked Tensor as PNG Images -----
def save_stacked_tensor_as_png(worker_id, frame_count, tensor):
    """
    Saves the stacked tensor to a folder structured as:
      tensor_stacks/worker_{worker_id}/stack_{frame_count}/
    Each of the 4 slices in the tensor is saved as a separate PNG image.
    """
    base_folder = "tensor_stacks"
    worker_folder = os.path.join(base_folder, f"worker_{worker_id}")
    os.makedirs(worker_folder, exist_ok=True)
    stack_folder = os.path.join(worker_folder, f"stack_{frame_count}")
    os.makedirs(stack_folder, exist_ok=True)
    
    # Save each slice (channel) as a separate image.
    for idx in range(4):
        # Convert the tensor slice to a NumPy array.
        img = tensor[idx].cpu().numpy()
        file_path = os.path.join(stack_folder, f"image_{idx}.png")
        plt.imsave(file_path, img, cmap='gray')
        print(f"Saved image for worker {worker_id}, frame {frame_count}, channel {idx} at {file_path}")

# ----- Paths and Dolphin Launcher -----
paths = {
    "dolphin_path": r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe",
    "script_path": r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\env_multi.py",
    "savestate_path": r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav",
    "game_path": r"E:\Games\Dolphin Games\MarioKart(Compress).iso",
}

def launch_dolphin_for_worker(worker_id):
    shm_name = f"dolphin_shared_{worker_id}"
    os.environ["SHM_NAME"] = shm_name

    # Use a unique user directory for each Dolphin instance:
    user_dir = f"C:\\Users\\Zachary\\DolphinUserDirs\\instance_{worker_id}"
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
    return shm_name

# ----- Global Replay Buffer -----
class GlobalReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = mp.Manager().list()  # shared list among processes
        self.lock = mp.Lock()

    def store(self, transition):
        with self.lock:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size):
        with self.lock:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
        return samples

# ----- Checkpoint Functions -----
def save_checkpoint(agent, optimizer, update_count, checkpoint_path):
    checkpoint = {
        'online_state_dict': agent.online_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'update_count': update_count
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at update {update_count} to {checkpoint_path}")

def load_checkpoint(agent, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.online_net.load_state_dict(checkpoint['online_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        update_count = checkpoint.get('update_count', 0)
        print(f"Checkpoint loaded from {checkpoint_path} at update {update_count}")
        return update_count
    else:
        print("No checkpoint found, starting fresh.")
        return 0

# ----- Worker Process -----
def worker_process(worker_id, global_buffer, global_weights, episode_rewards):
    print(f"Worker {worker_id}: Starting.")
    shm_name = launch_dolphin_for_worker(worker_id)
    try:
        shm = wait_for_shared_memory(shm_name, timeout=60)
    except TimeoutError as e:
        print(f"Worker {worker_id}: {e}. Exiting worker process.")
        return  # Exit gracefully

    shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)

    agent = RainbowDQN(
        state_shape=(4, 128, 128),
        num_actions=14,
        lr=1e-4,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        v_min=-5.0,
        v_max=1000.0,
        num_atoms=51,
        target_update_interval=2000,
        device='cuda'
    )
    if 'state_dict' in global_weights:
        agent.online_net.load_state_dict(global_weights['state_dict'])
        agent.online_net.to('cuda')

    local_frame_count = 0
    max_steps_per_episode = 1000

    while True:
        episode_reward = 0.0
        steps = 0
        done = False

        try:
            while not done and steps < max_steps_per_episode:
                state_np, reward, terminal, speed, lap_progress = read_shared_state(shm_array)
                # Preprocess state and obtain the stacked tensor.
                obs_tensor = preprocess_state(state_np)
                # Uncomment the line below if you want to save PNG images.
                # save_stacked_tensor_as_png(worker_id, local_frame_count, obs_tensor)
                
                action = agent.select_action(obs_tensor)
                write_action(shm_array, action)

                next_state_np, _, terminal, _, _ = read_shared_state(shm_array)
                next_obs_tensor = preprocess_state(next_state_np)

                transition = (
                    obs_tensor.cpu().numpy(),
                    action,
                    reward,
                    next_obs_tensor.cpu().numpy(),
                    terminal
                )
                global_buffer.store(transition)

                local_frame_count += 1
                steps += 1
                episode_reward += reward

                if local_frame_count % 1000 == 0 and 'state_dict' in global_weights:
                    agent.online_net.load_state_dict(global_weights['state_dict'])
                    agent.online_net.to('cuda')

                if terminal > 0:
                    print(f"Worker {worker_id}: Episode ended after {steps} steps. Total Reward = {episode_reward:.3f}")
                    done = True

            # At the end of the episode, print the cumulated reward
            print(f"Worker {worker_id}: Cumulative reward for this episode: {episode_reward:.3f}")
            # Record episode reward in the reward log.
            episode_rewards.append(episode_reward)
            time.sleep(0.1)
        except Exception as e:
            print(f"Worker {worker_id}: Exception occurred: {e}")
            time.sleep(1)

# ----- Master Trainer Process -----
def master_trainer(global_buffer, global_weights, loss_logs, num_updates=10000, batch_size=64, update_interval=2000, checkpoint_path='master_checkpoint.pt', checkpoint_interval=1000):
    print("Master trainer: Initializing master network on CUDA.")
    master_agent = RainbowDQN(
        state_shape=(4, 128, 128),
        num_actions=14,
        lr=1e-4,
        buffer_size=50000,
        batch_size=batch_size,
        gamma=0.99,
        v_min=-5.0,
        v_max=1000.0,
        num_atoms=51,
        target_update_interval=update_interval,
        device='cuda'
    )
    optimizer = master_agent.optimizer

    update_count = load_checkpoint(master_agent, optimizer, checkpoint_path)

    with torch.no_grad():
        cpu_state_dict = {k: v.cpu() for k, v in master_agent.online_net.state_dict().items()}
    global_weights['state_dict'] = cpu_state_dict

    while update_count < num_updates:
        if len(global_buffer.buffer) < batch_size:
            time.sleep(0.05)
            continue

        samples = global_buffer.sample(batch_size)
        obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)
        obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0) for o in obs_batch]).float().to('cuda')
        next_obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0) for o in next_obs_batch]).float().to('cuda')
        actions = torch.LongTensor(actions).unsqueeze(1).to('cuda')
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to('cuda')
        dones = torch.FloatTensor(dones).unsqueeze(1).to('cuda')

        master_agent.online_net.reset_noise()
        master_agent.target_net.reset_noise()

        dist = master_agent.online_net(obs_batch)
        dist = torch.log_softmax(dist, dim=2)
        dist_action = dist.gather(1, actions.unsqueeze(2).expand(batch_size, 1, master_agent.num_atoms)).squeeze(1)

        with torch.no_grad():
            next_dist = master_agent.online_net(next_obs_batch)
            next_dist = torch.softmax(next_dist, dim=2)
            q_values_next = torch.sum(next_dist * master_agent.support.view(1, 1, -1), dim=2)
            best_actions = q_values_next.argmax(dim=1)
            target_dist = master_agent.target_net(next_obs_batch)
            target_dist = torch.softmax(target_dist, dim=2)
            target_dist = target_dist[range(batch_size), best_actions]

        Tz = rewards + (1.0 - dones) * master_agent.gamma * master_agent.support.view(1, -1)
        Tz = Tz.clamp(master_agent.v_min, master_agent.v_max)
        b = (Tz - master_agent.v_min) / master_agent.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * master_agent.num_atoms, batch_size).long().unsqueeze(1).to('cuda')
        target_dist_projected = torch.zeros_like(target_dist).to('cuda')
        target_dist_projected.view(-1).index_add_(0, (l + offset).view(-1),
                                                   (target_dist * (u.float() - b)).view(-1))
        target_dist_projected.view(-1).index_add_(0, (u + offset).view(-1),
                                                   (target_dist * (b - l.float())).view(-1))

        losses = -torch.sum(target_dist_projected * dist_action, dim=1)
        loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_count += 1

        if update_count % 100 == 0:
            loss_val = loss.item()
            loss_logs.append((update_count, loss_val))
            print(f"Master update {update_count}: Loss = {loss_val:.4f}")

        if update_count % master_agent.target_update_interval == 0:
            master_agent.target_net.load_state_dict(master_agent.online_net.state_dict())

        with torch.no_grad():
            cpu_state_dict = {k: v.cpu() for k, v in master_agent.online_net.state_dict().items()}
        global_weights['state_dict'] = cpu_state_dict

        if update_count % checkpoint_interval == 0:
            save_checkpoint(master_agent, optimizer, update_count, checkpoint_path)
    
    print("Master trainer: Training finished.")

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
        print("Loss graph saved as loss_graph.png")
    else:
        print("No loss logs to plot.")

    if episode_rewards:
        plt.figure()
        plt.plot(range(len(episode_rewards)), episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Reward Over Time")
        plt.legend()
        plt.savefig("reward_graph.png")
        plt.close()
        print("Reward graph saved as reward_graph.png")
    else:
        print("No episode rewards to plot.")

# ----- Main Multi-Agent Launch -----
def main():
    num_workers = 8  # Number of parallel Dolphin/agent instances.
    global_buffer = GlobalReplayBuffer(capacity=50000)
    manager = mp.Manager()
    global_weights = manager.dict()
    loss_logs = manager.list()
    episode_rewards = manager.list()

    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, global_buffer, global_weights, episode_rewards))
        p.start()
        workers.append(p)

    try:
        master_trainer(global_buffer, global_weights, loss_logs, num_updates=1000000, batch_size=64)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Master trainer encountered an error: {e}")

    for p in workers:
        p.terminate()
        p.join()

    plot_metrics(list(loss_logs), list(episode_rewards))

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
