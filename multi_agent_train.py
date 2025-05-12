import os
import time
import torch
import numpy as np
import subprocess
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import concurrent.futures
from collections import deque
from agent import BTRAgent, choose_eval_action  # Using the new agent implementation
from multiprocessing.connection import Client
import collections

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)

def save_avg_reward_vs_frames(frames_history, avg_reward_history,
                              filepath="avg_reward_vs_frames.png"):
    """
    Plot running‑average episode reward against TOTAL frames processed.

    Parameters
    ----------
    frames_history : list[int]
        Cumulative frames processed at each snapshot (x‑coordinates).
    avg_reward_history : list[float]
        Running average episode reward at the same snapshots (y‑coordinates).
    """
    if not frames_history or len(frames_history) != len(avg_reward_history):
        logging.warning("Frame / reward history length mismatch – plot skipped.")
        return

    plt.figure()
    plt.plot(frames_history, avg_reward_history, marker='o')
    plt.xlabel("Total Frames Processed")
    plt.ylabel("Average Episode Reward")
    plt.title("Training Progress")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    logging.info("Progress plot saved to %s", filepath)

# --- Persistent Client for a Long-Lived Connection ---
class PersistentClient:
    def __init__(self, address, authkey=b'secret', timeout=12.0, retries=10, delay=1.0):
        self.address = address
        self.authkey = authkey
        self.timeout = timeout
        self.retries = retries
        self.delay = delay
        self.conn = None
        self.connect()
    
    def connect(self):
        attempt = 0
        while attempt < self.retries:
            try:
                self.conn = Client(self.address, authkey=self.authkey)
                return  # Successful connection.
            except ConnectionRefusedError:
                attempt += 1
                time.sleep(self.delay)
        raise ConnectionRefusedError(f"Could not connect to {self.address} after {self.retries} attempts")
    
    def send(self, data):
        try:
            self.conn.send(data)
            return self.conn.recv()
        except Exception as e:
            logging.error(f"Error during send/receive, reconnecting: {e}")
            self.connect()
            self.conn.send(data)
            return self.conn.recv()
    
    def close(self):
        if self.conn:
            self.conn.close()

# --- Launching Dolphin for Each Worker ---
def launch_dolphin_for_worker(worker_id):
    port = 6000 + worker_id
    os.environ["ENV_PORT"] = str(port)
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
    if user == "Nolan":
        paths["dolphin_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\dolphin-x64-framedrawn-stable\Dolphin.exe"
        paths["script_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\env_multi.py"
        paths["savestate_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\funky_flame_delfino.sav"
        paths["game_path"] = r"C:\Users\nolan\source\repos\dolphin\Source\Core\DolphinQt\MarioKart(Compress).iso"
    elif user == "Zach":
        paths["dolphin_path"] = r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe"
        paths["script_path"] = r"F:\MKWii_Capstone_Project\UPDATED_MKWii_Capstone\Mario-Kart-Wii-AI\env_multi.py"
        paths["savestate_path"] = r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
        paths["game_path"] = r"E:\Games\Dolphin Games\MarioKart(Compress).iso"
    elif user == "Victor":
        paths["dolphin_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\Dolphin.exe"
        paths["script_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\env_multi.py"
        paths["savestate_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\funky_flame_delfino.sav"
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
    return ('localhost', port), proc

# --- Vectorized Environment Using Persistent Connections ---
class VecDolphinEnv:
    def __init__(self, num_envs, frame_skip=4):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.env_addresses = []
        self.persistent_clients = []
        self.process_handles = []
        self.current_obs = []  # Latest observation per environment.
        for i in range(num_envs):
            address, proc = launch_dolphin_for_worker(i)
            self.env_addresses.append(address)
            self.process_handles.append(proc)
            client = PersistentClient(address)
            self.persistent_clients.append(client)
        for i, client in enumerate(self.persistent_clients):
            reset_msg = client.conn.recv()
            if isinstance(reset_msg, dict) and reset_msg.get("command") == "reset":
                self.current_obs.append(reset_msg["observation"])
                logging.info(f"Connected to environment at {self.env_addresses[i]} with initial observation.")
            else:
                self.current_obs.append(None)
    
    def restart_worker(self, i):
        logging.error(f"Worker {i} not responding. Restarting worker and Dolphin process.")
        try:
            self.process_handles[i].terminate()
            self.process_handles[i].wait(timeout=15)
        except Exception as e:
            logging.error(f"Error terminating worker {i}: {e}")
        address, proc = launch_dolphin_for_worker(i)
        self.env_addresses[i] = address
        self.process_handles[i] = proc
        self.persistent_clients[i].close()
        self.persistent_clients[i] = PersistentClient(address)
        reset_msg = self.persistent_clients[i].conn.recv()
        if isinstance(reset_msg, dict) and reset_msg.get("command") == "reset":
            self.current_obs[i] = reset_msg["observation"]
        else:
            self.current_obs[i] = None

    # Updated step function: converts using the existing NumPy array without extra wrapping.
    def step(self, actions):
        """Return next_obs, rewards, terminals, episode_rewards."""
        transitions = [None] * self.num_envs
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.persistent_clients[i].send,
                    {"command": "step", "action": actions[i]}
                )
                for i in range(self.num_envs)
            ]
            for i, future in enumerate(futures):
                transitions[i] = future.result()

        next_obs, rewards, terminals, episode_rewards = [], [], [], []
        for i, trans in enumerate(transitions):
            self.current_obs[i] = trans["next_observation"]
            next_obs.append(trans["next_observation"])
            rewards.append(trans["reward"])
            terminals.append(trans["terminal"])
            episode_rewards.append(trans["episode_rewards"] if trans["terminal"] else None)

        return next_obs, rewards, terminals, episode_rewards

    def close(self):
        for proc in self.process_handles:
            try:
                proc.terminate()
                proc.wait(timeout=15)
                logging.info("Terminated Dolphin process.")
            except Exception as e:
                logging.error(f"Error terminating Dolphin process: {e}")

checkpoint_path = "vectorized_checkpoint.pt"

def save_checkpoint(agent, update_count):
    checkpoint = {
        'online_state_dict': agent.net.state_dict(),
        'target_state_dict': agent.tgt_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'update_count': update_count
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at update count {update_count} to {checkpoint_path}")

def load_checkpoint(agent):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.net.load_state_dict(checkpoint['online_state_dict'])
        agent.tgt_net.load_state_dict(checkpoint['target_state_dict'])
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

# -------------------------------
# NEW: Evaluation Function
# -------------------------------
def evaluate_agent(agent, eval_client, num_episodes=100, frameskip=4):
    """
    Evaluates the current agent by running 'num_episodes' evaluation episodes.
    Evaluation epsilon is set to 0.01 until total environment frames reach 125M, then 0.
    """
    try:
        # Prepare evaluation network and disable noise.
        agent.prep_evaluation()  # This creates agent.eval_net and disables noise.
        
        # Compute evaluation epsilon based on total environment frames.
        env_frames = agent.env_steps * frameskip
        eval_epsilon = 0.01 if env_frames < 125e6 else 0.0
        logging.info(f"Evaluation Phase: env_frames={env_frames}, using eval_epsilon={eval_epsilon}")
        
        episode_rewards = []
        for ep in range(num_episodes):
            try:
                # Reset the evaluation environment.
                eval_client.send({"command": "reset"})
                response = eval_client.conn.recv()
                obs = response["observation"]
                done = False
                ep_reward = 0.0
                
                while not done:
                    # Use the evaluation network with fixed epsilon.
                    try:
                        action_tensor = choose_eval_action(
                            observation=obs,
                            eval_net=agent.eval_net,
                            n_actions=agent.n_actions,
                            device=agent.device,
                            rng=eval_epsilon
                        )
                        action = int(action_tensor.item() if isinstance(action_tensor, torch.Tensor) else action_tensor)
                    except Exception as e:
                        logging.error(f"Error in evaluation action selection: {e}")
                        # Default to a random action as fallback
                        action = np.random.randint(0, agent.n_actions)
                    
                    try:
                        eval_client.send({"command": "step", "action": action})
                        step_response = eval_client.conn.recv()
                        obs = step_response["next_observation"]
                        reward = step_response["reward"]
                        done = step_response["terminal"]
                        ep_reward += reward
                    except Exception as e:
                        logging.error(f"Error communicating with environment: {e}")
                        # Skip the rest of this episode
                        break
                
                episode_rewards.append(ep_reward)
                logging.info(f"Eval Episode {ep+1}/{num_episodes}: Total Reward = {ep_reward}")
            except Exception as e:
                logging.error(f"Error in evaluation episode {ep}: {e}")
                continue
        
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            logging.info(f"Evaluation complete: Average Reward over {len(episode_rewards)} episodes = {avg_reward}")
            return avg_reward
        else:
            logging.error("No valid evaluation episodes completed")
            return 0.0
    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        return 0.0

def main():
    total_steps = 0
    num_envs = 1
    debug_mode = False
    print_frames = False
    episode_rewards_log = []                # all finished‑episode returns
    frames_history       = []               # x‑coords for hourly snapshots
    avg_reward_history   = []               # y‑coords for hourly snapshots
    last_plot_time       = time.time()      # hourly timer
    loss_logs = []                          # Initialize loss_logs
    episode_rewards = []                    # Initialize episode_rewards list
    env = None                              # Initialize env for safety
    try:
        env = VecDolphinEnv(num_envs, frame_skip=4)
        obs = env.current_obs
        
        agent = BTRAgent(
            n_actions=6,
            input_dims=(4, 128, 128),  # Ensure these match the target resolution in env_multi.py.
            device='cuda',
            num_envs=num_envs,
            agent_name='BTRAgent',
            total_frames=200000000,
            testing=False,
            replay_period=64,
            per_beta_anneal=True,
            loading_checkpoint=True,
        )

        log_frequency = 1024
        
        logging.info("Using standard feed-forward BTRAgent with PER buffer")

        # NEW: Set next evaluation threshold in environment steps.
        next_eval_steps = 450000  # Every 250K env steps = 1M frames (if frameskip=4)
        
        # For evaluation, select a dedicated persistent client.
        # Here we choose the first client (agent 0) to perform evaluation.
        eval_client = env.persistent_clients[0]

        while True:
            total_steps += num_envs
            obs_tensor = torch.from_numpy(np.copy(obs)).to(agent.device).float()
            if total_steps >= 4000:
                actions = agent.choose_action(obs_tensor, debug=debug_mode)
            else:
                actions = agent.choose_action(obs_tensor)
            next_obs, rewards, terminals, epi_rewards = env.step(actions.numpy())
            for i in range(num_envs):
                # Store transitions using the original numpy observations.
                agent.store_transition(obs[i], actions[i].item(), rewards[i], next_obs[i], terminals[i], i)
                if epi_rewards[i] is not None:
                    episode_rewards_log.append(epi_rewards[i])
            obs = next_obs

            #--- Debug Saving ---
            if print_frames:
                debug_dir = "debug_data_after_store"
                os.makedirs(debug_dir, exist_ok=True)
                step_folder = os.path.join(debug_dir, f"step_{total_steps}")
                os.makedirs(step_folder, exist_ok=True)
                for worker in range(num_envs):
                    worker_folder = os.path.join(step_folder, f"worker_{worker}")
                    os.makedirs(worker_folder, exist_ok=True)
                    obs_stack = obs[worker].clone().cpu().squeeze(0)
                    next_stack = next_obs[worker].clone().cpu()
                    for frame_idx in range(obs_stack.shape[0]):
                        obs_filename = os.path.join(worker_folder, f"worker_{worker}_obs_frame_{frame_idx}_step_{total_steps}.png")
                        plt.imsave(obs_filename, obs_stack[frame_idx].numpy(), cmap='gray')
                        next_filename = os.path.join(worker_folder, f"worker_{worker}_next_frame_{frame_idx}_step_{total_steps}.png")
                        plt.imsave(next_filename, next_stack[frame_idx].numpy(), cmap='gray')
                #--- End Debug Saving ---
            
            if agent.memory.capacity % log_frequency == 0 and agent.memory.capacity < 1048576:
                if agent.memory.capacity < agent.min_sampling_size:
                    logging.info(f"Burn-in: {(agent.memory.capacity / agent.min_sampling_size) * 100}% ({agent.memory.capacity})")
                else:
                    logging.info(f"Buffer size = {agent.memory.capacity}")
            if agent.env_steps % 64 == 0:
                agent.learn()

            # ------------- hourly plot update -------------------------------
            if (time.time() - last_plot_time) >= 300 and episode_rewards_log:
                total_frames = total_steps * env.frame_skip
                avg_reward   = np.mean(episode_rewards_log)

                frames_history.append(total_frames)
                avg_reward_history.append(avg_reward)

                save_avg_reward_vs_frames(frames_history, avg_reward_history)

                episode_rewards_log.clear()
                last_plot_time = time.time()

            # NEW: Trigger evaluation every 250K environment steps.
            # if agent.env_steps >= next_eval_steps:
            #     logging.info(f"Triggering evaluation at env_steps = {agent.env_steps} "
            #                 f"({agent.env_steps * env.frame_skip} total frames)")
            #     avg_eval_reward = evaluate_agent(agent, eval_client, num_episodes=100, frameskip=env.frame_skip)
            #     logging.info(f"Evaluation Result at {agent.env_steps * env.frame_skip} frames: Average Reward = {avg_eval_reward}")
            #     next_eval_steps += 250000  # Update next evaluation threshold.

            if total_steps % (num_envs * 100000) == 0:
                # avg_reward = np.mean(rewards)
                # min_reward = np.min(rewards)
                # max_reward = np.max(rewards)
                # logging.info(f"Step {total_steps}: Avg Reward {avg_reward}, Min Reward {min_reward}, Max Reward {max_reward}")
                # Save checkpoint after sufficient experiences.
                if agent.memory.capacity >= 200000 and not agent.loading_checkpoint:
                    agent.save_model()
                elif agent.memory.capacity >= 300000 and agent.loading_checkpoint:
                    agent.save_model()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Exiting...")
        try:
            subprocess.run('taskkill /F /IM Dolphin.exe', check=True, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Dolphin instances closed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error closing Dolphin instances:", e.stderr)
    finally:
        plot_metrics(loss_logs, episode_rewards)
        if env is not None:
            env.close()

if __name__ == "__main__":
    main()
