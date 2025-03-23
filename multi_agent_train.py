import os
import time
import torch
import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
from collections import deque
from agent import BTRAgent  # your agent using PrioritizedReplayBuffer
from multiprocessing import shared_memory

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
    Blocks until a new timestep is detected.
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
        frame_resized = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        frames.append(frame_resized)
    stacked = torch.cat(frames, dim=0)  # Shape: (4, 128, 128)
    return stacked

def write_action(shm_array, action):
    """Writes the selected action to shared memory (column 2 of row 0)."""
    shm_array[0, 2] = action

def launch_dolphin_for_worker(worker_id):
    """
    Launches a Dolphin instance for a given worker ID. Each instance uses its own shared memory name.
    """
    shm_name = f"dolphin_shared_{worker_id}"
    os.environ["SHM_NAME"] = shm_name
    # Adjust user and paths as needed:
    user = "Zach"
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
    cmd = (
        f'"{paths["dolphin_path"]}" '
        f'-u "{user_dir}" '
        f'--no-python-subinterpreters '
        f'--script "{paths["script_path"]}" '
        f'--save_state="{paths["savestate_path"]}" '
        f'--exec="{paths["game_path"]}"'
    )
    subprocess.Popen(cmd, shell=True)
    logging.info(f"Vectorized Env {worker_id}: Launched Dolphin with command: {cmd}")
    return shm_name

# ----- Vectorized Environment Wrapper with Frame Skipping -----
class VecDolphinEnv:
    def __init__(self, num_envs, frame_skip=4):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.shm_arrays = []
        self.frame_buffers = []
        self.env_shms = []
        for i in range(num_envs):
            shm_name = launch_dolphin_for_worker(i)
            shm = wait_for_shared_memory(shm_name)
            self.env_shms.append(shm)
            shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)
            self.shm_arrays.append(shm_array)
            # Initialize frame buffer using the first observed state.
            initial_frame, _, _, _, _ = read_shared_state(shm_array)
            fb = deque(maxlen=4)
            for _ in range(4):
                fb.append(initial_frame)
            self.frame_buffers.append(fb)

    def step(self, actions):
        """
        For each environment, repeats the provided action for self.frame_skip frames.
        Accumulates rewards and checks for terminal flags.
        Returns:
          next_obs (list of tensors), total_rewards (list of floats), terminals (list of 0/1 flags)
        """
        next_obs = []
        total_rewards = []
        terminals = []
        for i in range(self.num_envs):
            acc_reward = 0.0
            term_flag = 0
            # For frame skipping: repeat action for self.frame_skip frames
            for _ in range(self.frame_skip):
                write_action(self.shm_arrays[i], actions[i])
                state, reward, terminal, speed, lap_progress = read_shared_state(self.shm_arrays[i])
                self.frame_buffers[i].append(state)
                acc_reward += reward
                if terminal > 0:
                    term_flag = 1
                    break  # End frame skip early if terminal reached
            total_rewards.append(acc_reward)
            terminals.append(term_flag)
            obs_tensor = preprocess_frame_stack(list(self.frame_buffers[i]))
            next_obs.append(obs_tensor)
        return next_obs, total_rewards, terminals

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

def update_agent_and_log_loss(agent):
    """Re-implements the update step (using agent.buffer which is the prioritized replay buffer)
    and returns the loss value for logging."""
    if len(agent.buffer.buffer) < agent.batch_size:
        return None
    agent.online_net.train()
    agent.target_net.train()
    samples, indices, weights = agent.buffer.sample(agent.batch_size, beta=0.4)
    weights = weights.to(agent.device)
    obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)
    obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in obs_batch]).to(agent.device)
    next_obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in next_obs_batch]).to(agent.device)
    actions = torch.LongTensor(actions).to(agent.device)
    rewards = torch.FloatTensor(rewards).to(agent.device)
    dones = torch.FloatTensor(dones).to(agent.device)
    quantiles, taus = agent.online_net(obs_batch)
    actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, agent.online_net.num_quantiles, 1)
    current_quantiles = quantiles.gather(2, actions_expanded).squeeze(2)
    q_values = quantiles.mean(dim=1)
    logits = q_values / agent.munchausen_tau
    log_policy = torch.log_softmax(logits, dim=1)
    munchausen_bonus = torch.clamp(log_policy.gather(1, actions.unsqueeze(1)), min=agent.munchausen_clip)
    rewards_aug = rewards + agent.munchausen_alpha * munchausen_bonus.squeeze(1)
    next_quantiles_online, _ = agent.online_net(next_obs_batch)
    next_q_values_online = next_quantiles_online.mean(dim=1)
    best_actions = next_q_values_online.argmax(dim=1)
    best_actions_expanded = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, agent.online_net.num_quantiles, 1)
    next_quantiles_target, _ = agent.target_net(next_obs_batch)
    next_quantiles = next_quantiles_target.gather(2, best_actions_expanded).squeeze(2)
    target_quantiles = rewards_aug.unsqueeze(1) + agent.gamma * next_quantiles * (1 - dones.unsqueeze(1))
    target_quantiles = target_quantiles.detach()
    taus = taus.unsqueeze(2)
    target_expanded = target_quantiles.unsqueeze(1)
    td_error = target_expanded - current_quantiles.unsqueeze(2)
    huber_loss = agent._huber_loss(td_error, k=1.0)
    quantile_loss = (torch.abs(taus - (td_error.detach() < 0).float()) * huber_loss) / 1.0
    # Compute per-sample losses (shape: [batch_size])
    sample_losses = quantile_loss.mean(dim=2).sum(dim=1)

    # Compute overall loss as a weighted average
    loss = (sample_losses * weights.squeeze()).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # Use the per-sample losses for new priorities
    new_priorities = sample_losses.detach().abs().cpu() + 1e-6
    agent.buffer.update_priorities(indices, new_priorities)

    agent.update_count += 1
    if agent.update_count % agent.target_update_interval == 0:
        logging.info(f"Updating target net at update {agent.update_count}")
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        save_checkpoint(agent, agent.update_count)
    return loss.item()

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

# ----- Main Training Loop -----
def main():
    num_envs = 4  # Number of parallel Dolphin instances
    env = VecDolphinEnv(num_envs, frame_skip=4)
    agent = BTRAgent(
        state_shape=(4, 128, 128),
        num_actions=8,
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
    # Load checkpoint if available
    load_checkpoint(agent)
    total_steps = 0
    update_frequency = 50  # update every 50 steps
    loss_logs = []       # (update count, loss)
    episode_rewards = [] # list of episode total rewards

    try:
        while True:
            # 1. For each environment, get the current observation from its frame buffer.
            obs_list = []
            for i in range(num_envs):
                obs_tensor = preprocess_frame_stack(list(env.frame_buffers[i]))
                obs_list.append(obs_tensor.unsqueeze(0))
            batch_obs = torch.cat(obs_list, dim=0).to('cuda')  # Shape: (num_envs, 4, 128, 128)

            # 2. Select actions for all environments via a batched forward pass.
            with torch.no_grad():
                quantiles, taus = agent.online_net(batch_obs)
                q_mean = quantiles.mean(dim=1)
                actions = q_mean.argmax(dim=1).cpu().numpy().tolist()

            # 3. Step all environments with the chosen actions (frame skipping handled inside).
            next_obs_list, rewards, terminals = env.step(actions)

            # 4. For each environment, store the transition using the agent's prioritized replay buffer.
            for i in range(num_envs):
                transition = (
                    batch_obs[i].cpu().numpy().astype(np.uint8),  # current state
                    actions[i],
                    rewards[i],
                    next_obs_list[i].cpu().numpy().astype(np.uint8),  # next state
                    terminals[i]
                )
                agent.store_transition(transition)
            total_steps += num_envs

            # 5. Periodically update the agent and log loss.
            loss_val = update_agent_and_log_loss(agent)
            if loss_val is not None:
                loss_logs.append((agent.update_count, loss_val))
                logging.info(f"Update {agent.update_count}: Loss = {loss_val:.4f}")

            # 6. If any environment signals episode termination, log reward and checkpoint.
            for term, rew in zip(terminals, rewards):
                if term > 0:
                    episode_rewards.append(rew)
                    logging.info(f"Episode ended with reward: {rew:.3f}")
                    save_checkpoint(agent, agent.update_count)

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

if __name__ == "__main__":
    main()
