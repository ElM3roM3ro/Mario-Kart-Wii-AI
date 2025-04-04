import os
import time
import torch
import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
from collections import deque
from agent import BTRAgent  # Now using the new agent implementation
from gym.wrappers import LazyFrames
import collections
from multiprocessing.connection import Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)

Ymem = 128
Xmem = 128
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_FRAMES = 2_000_000
EPSILON_DISABLED_FRAMES = 100_000_000

total_frames = 0

import concurrent.futures

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
                return  # Successful connection, exit the loop.
            except ConnectionRefusedError as e:
                attempt += 1
                time.sleep(self.delay)
        # If all attempts fail, raise the error.
        raise ConnectionRefusedError(f"Could not connect to {self.address} after {self.retries} attempts")
    
    def send(self, data):
        try:
            self.conn.send(data)
            return self.conn.recv()
        except Exception as e:
            # Reconnect and retry on error.
            self.connect()
            self.conn.send(data)
            return self.conn.recv()
    
    def close(self):
        if self.conn:
            self.conn.close()

def set_env_action(address, action, authkey=b'secret'):
    try:
        conn = Client(address, authkey=authkey)
        conn.send({"command": "set_action", "value": action})
        response = conn.recv()
        conn.close()
        return response
    except Exception as e:
        logging.error(f"Error setting action at {address}: {e}")
        return None

def get_env_state(address, authkey=b'secret', timeout=12.0):
    start_time = time.time()
    while True:
        try:
            conn = Client(address, authkey=authkey)
            conn.send({"command": "get_state"})
            state = conn.recv()
            conn.close()
            return state
        except Exception as e:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout getting state from {address}: {e}")
            logging.info(f"Waiting for environment at {address} to be ready...")
            time.sleep(1)

def preprocess_frame_stack(frame_stack):
    frames = np.stack(frame_stack, axis=0)  # shape: (framestack, H, W)
    tensor = torch.from_numpy(frames).float().to('cuda')
    return tensor

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
    return ('localhost', port), proc

class VecDolphinEnv:
    def __init__(self, num_envs, frame_skip=4):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.env_addresses = []
        self.persistent_clients = []
        self.frame_buffers = []
        self.process_handles = []
        for i in range(num_envs):
            address, proc = launch_dolphin_for_worker(i)
            self.env_addresses.append(address)
            self.process_handles.append(proc)
            # Create a persistent client per environment.
            self.persistent_clients.append(PersistentClient(address))
        for i, addr in enumerate(self.env_addresses):
            state = self.persistent_clients[i].send({"command": "get_state"})
            logging.info(f"Connected to environment at {addr} with timestep {state['timestep']}")
            fb = deque(maxlen=4)
            for _ in range(4):
                fb.append(state["frame"])
            self.frame_buffers.append(fb)
    
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
        state = get_env_state(address)
        fb = deque(maxlen=4)
        for _ in range(4):
            fb.append(state["frame"])
        self.frame_buffers[i] = fb

    def step(self, actions):
        global total_frames
        next_obs = [None] * self.num_envs
        total_rewards = [0.0] * self.num_envs
        terminals = [0] * self.num_envs

        # Set actions concurrently.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            set_futures = [executor.submit(self.persistent_clients[i].send,
                                             {"command": "set_action", "value": actions[i]})
                           for i in range(self.num_envs)]
            for future in concurrent.futures.as_completed(set_futures):
                _ = future.result()

        # Retrieve states concurrently.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            get_futures = [executor.submit(self.persistent_clients[i].send,
                                             {"command": "get_state"})
                           for i in range(self.num_envs)]
            results = [future.result() for future in get_futures]

        for i, state in enumerate(results):
            self.frame_buffers[i].append(state["frame"])
            if state["terminal"]:
                # Reset the frame buffer on terminal.
                fb = deque(maxlen=4)
                for _ in range(4):
                    fb.append(state["frame"])
                self.frame_buffers[i] = fb
            obs_tensor = preprocess_frame_stack(list(self.frame_buffers[i]))
            next_obs[i] = obs_tensor
            total_rewards[i] = state["reward"]
            terminals[i] = state["terminal"]
        
        total_frames += (4 * self.num_envs)
        if total_frames % 1000 == 0:
            logging.info(f"Frames = {total_frames}")
        return next_obs, total_rewards, terminals

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
    num_envs = 8
    buffer_size = 0
    try:
        env = VecDolphinEnv(num_envs, frame_skip=4)
        # Instantiate the new agent with updated defaults (128×128 images)
        agent = BTRAgent(
            n_actions=8,
            input_dims=(4, 128, 128),
            device='cuda',
            num_envs=num_envs,
            agent_name='BTRAgent',
            total_frames=0,
            testing=True,
            rr=0.015625
        )
        load_checkpoint(agent)
        loss_logs = []
        episode_rewards = []
        while True:
            current_states = [np.array(LazyFrames(list(env.frame_buffers[i]))) for i in range(num_envs)]
            obs_list = []
            for i in range(num_envs):
                obs_tensor = preprocess_frame_stack(list(env.frame_buffers[i]))
                obs_list.append(obs_tensor.unsqueeze(0))
            batch_obs = torch.cat(obs_list, dim=0).to('cuda')
            # Use the agent to choose actions (assuming agent.choose_action works with a batch)
            actions = agent.choose_action(batch_obs)
            next_obs, rewards, terminals = env.step(actions.numpy())
            # Store transitions and perform learning for each environment
            for i in range(num_envs):
                agent.store_transition(current_states[i], actions[i].item(), rewards[i], next_obs[i], terminals[i], i)
                buffer_size += 1
            if buffer_size % 1000 == 0:
                logging.info(f"Buffer size = {buffer_size}")
            agent.learn()
            total_steps += num_envs
            if total_steps % 1000 == 0:
                avg_reward = np.mean(rewards)  # average reward for the current step
                logging.info(f"Step {total_steps}: Total Frames {env.frame_skip * total_steps}, "
                            f"Env Steps {agent.env_steps}, Grad Steps {agent.grad_steps}, "
                            f"Epsilon {agent.epsilon.eps:.4f}, Avg Reward {avg_reward:.4f}")
                save_checkpoint(agent, total_steps)
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
