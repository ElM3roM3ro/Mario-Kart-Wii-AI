import os
import time
import torch
import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
import concurrent.futures
from collections import deque
from agent import BTRAgent  # Using the new agent implementation
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
        # Perform handshake: get initial observation from each env.
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

    # Updated step function: directly uses the transition returned by env_multi.py.
    def step(self, actions):
        transitions = [None] * self.num_envs
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.persistent_clients[i].send,
                                         {"command": "step", "action": actions[i]})
                       for i in range(self.num_envs)]
            for i, future in enumerate(futures):
                transitions[i] = future.result()
        next_obs = []
        rewards = []
        terminals = []
        for i, trans in enumerate(transitions):
            self.current_obs[i] = trans["next_observation"]
            obs_tensor = torch.from_numpy(np.array(trans["next_observation"])).float().to('cuda')
            next_obs.append(obs_tensor)
            rewards.append(trans["reward"])
            terminals.append(trans["terminal"])
        return next_obs, rewards, terminals

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

def main():
    total_steps = 0
    num_envs = 1
    try:
        env = VecDolphinEnv(num_envs, frame_skip=4)
        agent = BTRAgent(
            n_actions=8,
            input_dims=(4, 128, 128),  # Ensure these match the target resolution in env_multi.py.
            device='cuda',
            num_envs=num_envs,
            agent_name='BTRAgent',
            total_frames=0,
            testing=True
        )
        load_checkpoint(agent)
        loss_logs = []
        episode_rewards = []
        while True:
            batch_obs = []
            for i in range(num_envs):
                obs_tensor = torch.from_numpy(np.array(env.current_obs[i])).float().to('cuda')
                batch_obs.append(obs_tensor.unsqueeze(0))
            batch_obs = torch.cat(batch_obs, dim=0)
            actions = agent.choose_action(batch_obs)
            next_obs, rewards, terminals = env.step(actions.numpy())
            
            for i in range(num_envs):
                agent.store_transition(batch_obs[i].clone(), actions[i].item(), rewards[i], next_obs[i].clone(), terminals[i], i)
            # #--- Debug Saving: Save Frames After Transition (optional) ---
            # #(This section remains commented; uncomment for debugging purposes.)
            # debug_dir = "debug_data_after_store"
            # os.makedirs(debug_dir, exist_ok=True)
            # step_folder = os.path.join(debug_dir, f"step_{total_steps}")
            # os.makedirs(step_folder, exist_ok=True)
            # for worker in range(num_envs):
            #     worker_folder = os.path.join(step_folder, f"worker_{worker}")
            #     os.makedirs(worker_folder, exist_ok=True)
            #     obs_stack = batch_obs[worker].cpu().squeeze(0)
            #     next_stack = next_obs[worker].cpu()
            #     for frame_idx in range(obs_stack.shape[0]):
            #         obs_filename = os.path.join(worker_folder, f"worker_{worker}_obs_frame_{frame_idx}_step_{total_steps}.png")
            #         plt.imsave(obs_filename, obs_stack[frame_idx].numpy(), cmap='gray')
            #         next_filename = os.path.join(worker_folder, f"worker_{worker}_next_frame_{frame_idx}_step_{total_steps}.png")
            #         plt.imsave(next_filename, next_stack[frame_idx].numpy(), cmap='gray')
            # #--- End Debug Saving ---
            
            if agent.memory.capacity % 1000 == 0:
                logging.info(f"Buffer size = {agent.memory.capacity}")
            agent.learn()
            total_steps += num_envs
            if total_steps % 1000 == 0:
                avg_reward = np.mean(rewards)
                logging.info(f"Step {total_steps}: Avg Reward {avg_reward:.4f}")
                # Save checkpoint after sufficient experiences.
                if agent.memory.capacity >= 200000:
                    save_checkpoint(agent, total_steps)
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
