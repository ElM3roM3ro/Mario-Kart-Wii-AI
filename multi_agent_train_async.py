import os
import time
import torch
import numpy as np
import subprocess
import logging
import matplotlib.pyplot as plt
import concurrent.futures
import threading
import queue
from collections import deque
from agent_async import BTRAgent, choose_eval_action
from multiprocessing.connection import Client

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

def launch_dolphin_for_worker(worker_id):
    port = 6000 + worker_id
    os.environ["ENV_PORT"] = str(port)
    user = "Zach"  # Change to your username as needed
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

class AsyncVecDolphinEnv:
    def __init__(self, num_envs, frame_skip=4):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.env_addresses = []
        self.persistent_clients = []
        self.process_handles = []
        self.current_obs = []  # Latest observation per environment
        self.running = True
        
        # Queues for asynchronous communication
        self.action_queues = [queue.Queue() for _ in range(num_envs)]
        self.experience_queue = queue.Queue(maxsize=10000)  # Shared queue for all experiences
        self.episode_reward_queue = queue.Queue()
        
        # Stats tracking
        self.total_frames = 0
        self.worker_steps = [0] * num_envs
        
        # Start environments
        for i in range(num_envs):
            address, proc = launch_dolphin_for_worker(i)
            self.env_addresses.append(address)
            self.process_handles.append(proc)
            client = PersistentClient(address)
            self.persistent_clients.append(client)
            
        # Wait for initial observations
        for i, client in enumerate(self.persistent_clients):
            reset_msg = client.conn.recv()
            if isinstance(reset_msg, dict) and reset_msg.get("command") == "reset":
                self.current_obs.append(reset_msg["observation"])
                logging.info(f"Connected to environment at {self.env_addresses[i]} with initial observation.")
            else:
                self.current_obs.append(None)
        
        # Start worker threads
        self.worker_threads = []
        for i in range(num_envs):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
    
    def _worker_loop(self, worker_id):
        """Worker thread that runs independently for each environment"""
        client = self.persistent_clients[worker_id]
        obs = self.current_obs[worker_id]
        
        while self.running:
            try:
                # Wait for an action with timeout
                try:
                    action = self.action_queues[worker_id].get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Take a step in the environment
                response = client.send({"command": "step", "action": action})
                
                next_obs = response["next_observation"]
                reward = response["reward"]
                done = response["terminal"]
                
                # Store the experience in the shared queue
                experience = {
                    "state": obs,
                    "action": action,
                    "reward": reward,
                    "next_state": next_obs,
                    "done": done,
                    "worker_id": worker_id
                }
                self.experience_queue.put(experience)
                
                # Track episode rewards if episode is done
                if done and response.get("episode_rewards") is not None:
                    self.episode_reward_queue.put({
                        "worker_id": worker_id,
                        "episode_reward": response["episode_rewards"],
                        "step": self.worker_steps[worker_id]
                    })
                
                # Update current observation
                obs = next_obs
                self.current_obs[worker_id] = next_obs
                
                # Update step counter
                self.worker_steps[worker_id] += 1
                
            except Exception as e:
                logging.error(f"Error in worker {worker_id}: {e}")
                self._restart_worker(worker_id)
                obs = self.current_obs[worker_id]
    
    def _restart_worker(self, worker_id):
        """Restart a worker if it fails"""
        logging.error(f"Worker {worker_id} not responding. Restarting worker and Dolphin process.")
        try:
            self.process_handles[worker_id].terminate()
            self.process_handles[worker_id].wait(timeout=15)
        except Exception as e:
            logging.error(f"Error terminating worker {worker_id}: {e}")
        
        address, proc = launch_dolphin_for_worker(worker_id)
        self.env_addresses[worker_id] = address
        self.process_handles[worker_id] = proc
        self.persistent_clients[worker_id].close()
        self.persistent_clients[worker_id] = PersistentClient(address)
        
        reset_msg = self.persistent_clients[worker_id].conn.recv()
        if isinstance(reset_msg, dict) and reset_msg.get("command") == "reset":
            self.current_obs[worker_id] = reset_msg["observation"]
        else:
            self.current_obs[worker_id] = None
    
    def step_async(self, actions):
        """Send actions to environments without waiting"""
        for i, action in enumerate(actions):
            self.action_queues[i].put(action)
    
    def get_experiences(self, timeout=0.0):
        """Get all available experiences without blocking"""
        experiences = []
        start_time = time.time()
        
        while time.time() - start_time < timeout or timeout == 0:
            try:
                exp = self.experience_queue.get_nowait()
                experiences.append(exp)
                self.total_frames += 1
            except queue.Empty:
                break
        
        return experiences
    
    def get_episode_rewards(self):
        """Get all completed episode rewards"""
        rewards = []
        while True:
            try:
                reward_info = self.episode_reward_queue.get_nowait()
                rewards.append(reward_info)
            except queue.Empty:
                break
        return rewards
    
    def close(self):
        """Stop all workers and clean up"""
        self.running = False
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        for proc in self.process_handles:
            try:
                proc.terminate()
                proc.wait(timeout=15)
                logging.info("Terminated Dolphin process.")
            except Exception as e:
                logging.error(f"Error terminating Dolphin process: {e}")

class AsyncLearner:
    def __init__(self, agent, replay_period=64, min_experiences=200_000):
        self.agent           = agent
        self.replay_period   = replay_period
        self.min_experiences = min_experiences
        self.last_learn_at   = 0            # <- NEW
        self.running         = True

        self.learning_thread = threading.Thread(
            target=self._learning_loop, daemon=True)
        self.learning_thread.start()

    def _learning_loop(self):
        """Gradient‑step whenever 64 new environment steps have happened."""
        while self.running:
            env_steps = self.agent.env_steps      # single source of truth
            if (env_steps - self.last_learn_at) >= self.replay_period \
               and env_steps >= self.min_experiences:
                self.agent.learn()                # do the update
                self.last_learn_at = env_steps
            else:
                time.sleep(0.001)                 # short nap to yield CPU
    
    def close(self):
        """Stop the learning thread"""
        self.running = False
        self.learning_thread.join(timeout=2.0)

def evaluate_agent(agent, eval_client, num_episodes=100, frameskip=4):
    """
    Evaluates the current agent by running 'num_episodes' evaluation episodes.
    Evaluation epsilon is set to 0.01 until total environment frames reach 125M, then 0.
    """
    # Prepare evaluation network and disable noise.
    agent.prep_evaluation()  # This creates agent.eval_net and disables noise.
    
    # Compute evaluation epsilon based on total environment frames.
    env_frames = agent.env_steps * frameskip
    eval_epsilon = 0.01 if env_frames < 125e6 else 0.0
    logging.info(f"Evaluation Phase: env_frames={env_frames}, using eval_epsilon={eval_epsilon}")
    
    episode_rewards = []
    for ep in range(num_episodes):
        # Reset the evaluation environment.
        eval_client.send({"command": "reset"})
        response = eval_client.conn.recv()
        obs = response["observation"]
        done = False
        ep_reward = 0.0
        
        while not done:
            # Use the evaluation network with fixed epsilon.
            action_tensor = choose_eval_action(
                observation=obs,
                eval_net=agent.eval_net,
                n_actions=agent.n_actions,
                device=agent.device,
                rng=eval_epsilon
            )
            action = int(action_tensor.item() if isinstance(action_tensor, torch.Tensor) else action_tensor)
            eval_client.send({"command": "step", "action": action})
            step_response = eval_client.conn.recv()
            obs = step_response["next_observation"]
            reward = step_response["reward"]
            done = step_response["terminal"]
            ep_reward += reward
        episode_rewards.append(ep_reward)
        logging.info(f"Eval Episode {ep+1}/{num_episodes}: Total Reward = {ep_reward}")
    avg_reward = np.mean(episode_rewards)
    logging.info(f"Evaluation complete: Average Reward over {num_episodes} episodes = {avg_reward}")
    return avg_reward

def save_checkpoint(agent, update_count, path="vectorized_checkpoint.pt"):
    checkpoint = {
        'online_state_dict': agent.net.state_dict(),
        'target_state_dict': agent.tgt_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'update_count': update_count
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at update count {update_count} to {path}")

def load_checkpoint(agent, path="vectorized_checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        agent.net.load_state_dict(checkpoint['online_state_dict'])
        agent.tgt_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.update_count = checkpoint.get('update_count', 0)
        logging.info(f"Loaded checkpoint from {path}")

def main():
    num_envs = 4
    debug_mode = False
    print_frames = False
    episode_rewards_log = []  # all finished‑episode returns
    frames_history = []       # x‑coords for hourly snapshots
    avg_reward_history = []   # y‑coords for hourly snapshots
    last_plot_time = time.time()
    
    try:
        # Initialize the asynchronous environment
        env = AsyncVecDolphinEnv(num_envs, frame_skip=4)
        
        # Initialize the agent
        agent = BTRAgent(
            n_actions=6,
            input_dims=(4, 128, 128),  # Ensure these match the target resolution in env_multi.py
            device='cuda',
            num_envs=num_envs,
            agent_name='BTRAgent',
            total_frames=200000000,
            testing=False,
            replay_period=64,
            per_beta_anneal=True
        )
        
        # Load checkpoint if needed
        agent.loading_checkpoint = False
        if agent.loading_checkpoint:
            agent.load_models(agent.agent_name + ".model")
        
        # Initialize the asynchronous learner
        learner = AsyncLearner(agent, replay_period=64, min_experiences=200000 if not agent.testing else 4000)
        
        # For evaluation - select a dedicated client
        eval_client = env.persistent_clients[0]
        next_eval_steps = 450000  # Every 250K env steps = 1M frames (if frameskip=4)
        
        # Main loop
        last_checkpoint_time = time.time()
        observation_batch = [np.array(env.current_obs[i]) for i in range(num_envs)]
        
        # Initially get actions for all environments
        batch_obs = torch.tensor(np.array(observation_batch), dtype=torch.float).to('cuda')
        actions = agent.choose_action(batch_obs, debug=debug_mode).numpy()
        
        # Send initial actions asynchronously
        env.step_async(actions)
        
        while True:
            # Process available experiences
            experiences = env.get_experiences()
            
            if experiences:
                # Process each experience and store in the agent's replay buffer
                for exp in experiences:
                    state = exp["state"]
                    action = exp["action"]
                    reward = exp["reward"]
                    next_state = exp["next_state"]
                    done = exp["done"]
                    worker_id = exp["worker_id"]
                    
                    # Store in agent's replay buffer
                    agent.store_transition(state, action, reward, next_state, done, worker_id)
                
                # Get latest observations for all environments
                observation_batch = [np.array(env.current_obs[i]) for i in range(num_envs)]
                
                # Get new actions for all environments
                batch_obs = torch.tensor(np.array(observation_batch), dtype=torch.float).to('cuda')
                actions = agent.choose_action(batch_obs, debug=debug_mode).numpy()
                
                # Send actions asynchronously
                env.step_async(actions)
            
            # Process completed episode rewards
            episode_rewards_info = env.get_episode_rewards()
            for reward_info in episode_rewards_info:
                episode_rewards_log.append(reward_info["episode_reward"])
                #logging.info(f"Worker {reward_info['worker_id']} completed episode with reward: {reward_info['episode_reward']}")
            
            # Log progress
            if agent.memory.capacity % 1024 == 0 and agent.memory.capacity > 0 and agent.memory.capacity < 1045876:
                if agent.memory.capacity < 200000 and not agent.testing:
                    logging.info(f"Burn-in: {(agent.memory.capacity / 200000) * 100:.2f}% ({agent.memory.capacity})")
                elif agent.memory.capacity < 4000 and agent.testing:
                    logging.info(f"Burn-in: {(agent.memory.capacity / 4000) * 100:.2f}% ({agent.memory.capacity})")
                else:
                    logging.info(f"Buffer size = {agent.memory.capacity}")
            
            # Plot metrics at regular intervals
            if (time.time() - last_plot_time) >= 600 and episode_rewards_log:
                total_frames = agent.env_steps * env.frame_skip  # 4 frames per new env step
                avg_reward = np.mean(episode_rewards_log[-100:]) if len(episode_rewards_log) > 100 else np.mean(episode_rewards_log)
                
                frames_history.append(total_frames)
                avg_reward_history.append(avg_reward)
                
                save_avg_reward_vs_frames(frames_history, avg_reward_history)
                last_plot_time = time.time()
            
            # Save checkpoint periodically
            if (time.time() - last_checkpoint_time) >= 3600 and agent.memory.capacity >= 200000:  # Every hour
                agent.save_model()
                last_checkpoint_time = time.time()
            
            # Optional: Run evaluation periodically
            # if agent.env_steps >= next_eval_steps:
            #     logging.info(f"Triggering evaluation at env_steps = {agent.env_steps} "
            #                 f"({agent.env_steps * env.frame_skip} total frames)")
            #     avg_eval_reward = evaluate_agent(agent, eval_client, num_episodes=5, frameskip=env.frame_skip)
            #     logging.info(f"Evaluation Result at {agent.env_steps * env.frame_skip} frames: Average Reward = {avg_eval_reward}")
            #     next_eval_steps += 250000  # Update next evaluation threshold
            
            # Sleep a tiny bit to avoid CPU hogging
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Exiting...")
        try:
            subprocess.run('taskkill /F /IM Dolphin.exe', check=True, shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Dolphin instances closed successfully.")
        except subprocess.CalledProcessError as e:
            print("Error closing Dolphin instances:", e.stderr)
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        if 'learner' in locals():
            learner.close()
        
        # Save final metrics and model
        if episode_rewards_log:
            plt.figure()
            plt.plot(range(len(episode_rewards_log)), episode_rewards_log, label="Episode Reward")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Episode Reward Over Time")
            plt.legend()
            plt.savefig("reward_graph.png")
            plt.close()

if __name__ == "__main__":
    main()