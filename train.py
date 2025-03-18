import os
import time
import subprocess
import numpy as np
from multiprocessing import shared_memory
import torch
from agent import RainbowDQN

# Shared memory parameters – must match those in env.py
Ymem = 78
Xmem = 94  # Must include extra debug columns (col 5: speed, col 6: lap_progress)
shm_name = 'dolphin_shared'
data_shape = (Ymem + 1, Xmem)

paths = {
    "dolphin_path": r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\dolphin-x64-framedrawn-stable\Dolphin.exe",
    "savestate_path": r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav",
    "script_path": r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\env.py",
    "game_path": r"C:\Users\nolan\source\repos\dolphin\Source\Core\DolphinQt\MarioKart(Compress).iso",
}

user = "Victor"
if user == "Nolan":
    paths["dolphin_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\dolphin-x64-framedrawn-stable\Dolphin.exe"
    paths["script_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\env.py"
    paths["savestate_path"] = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
    paths["game_path"] = r"C:\Users\nolan\source\repos\dolphin\Source\Core\DolphinQt\MarioKart(Compress).iso"
elif user == "Zach":
    paths["dolphin_path"] = r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe"
    paths["script_path"] = r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\env.py"
    paths["savestate_path"] = r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
    paths["game_path"] = r"E:\Games\Dolphin Games\MarioKart(Compress).iso"
elif user == "Victor":
    paths["dolphin_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\Dolphin.exe"
    paths["script_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\env.py"
    paths["savestate_path"] = r"C:\Users\victo\FunkyKong\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav"
    paths["game_path"] = r"C:\Users\victo\FunkyKong\dolphin-x64-framedrawn-stable\MarioKart(Compress).iso"
def launch_dolphin():
    # Update these paths to match your system
    dolphin_path = paths["dolphin_path"]  # Path to Dolphin.exe
    script_path = paths["script_path"]       # Path to env.py script
    savestate_path = paths["savestate_path"]  # Savestate file at countdown
    game_path = paths["game_path"]                 # Your game file
    # Build the command string without the 'cd' part.
    cmd = (
        f'"{dolphin_path}" --no-python-subinterpreters '
        f'--script "{script_path}" '
        f'--save_state="{savestate_path}" '
        f'--exec="{game_path}"'
    )
    
    print("Launching Dolphin with command:")
    print(cmd)
    
    subprocess.Popen(cmd, shell=True)
    print("Launched Dolphin with savestate loaded.")
    
    # Increase wait time as needed for Dolphin to fully load the savestate
    time.sleep(10)

def wait_for_shared_memory():
    """Wait until the shared memory is created by env.py."""
    while True:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            print("Shared memory created.")
            return shm
        except FileNotFoundError:
            print("Waiting for shared memory to be created...")
            time.sleep(1)

def read_shared_state(shm_array):
    """
    Reads from shared memory:
      Row 0: [timestep, timestep, action, reward, terminal, speed, lap_progress]
      Rows 1: state as an image (uint8)
    Returns:
      state (NumPy array of shape (Ymem, Xmem)),
      reward (float),
      terminal (float),
      speed (float),
      lap_progress (float)
    """
    print("Reading shared memory...")
    t0 = shm_array[0, 0]
    timeout = 0
    while True:
        time.sleep(0.01)
        t = shm_array[0, 0]
        if t != t0:
            break
        timeout += 1
        if timeout % 1000 == 0:
            print(f"Still waiting in read_shared_state (t={t}, t0={t0})...")
    state = shm_array[1:, :].copy().astype(np.uint8)
    reward = float(shm_array[0, 3])
    terminal = float(shm_array[0, 4])
    speed = float(shm_array[0, 5])
    lap_progress = float(shm_array[0, 6])
    print(f"Shared state read: Reward = {reward}, Terminal = {terminal}, Speed = {speed}, Lap Progress = {lap_progress}")
    return state, reward, terminal, speed, lap_progress

def write_action(shm_array, action):
    print(f"Writing action: {action}")
    shm_array[0, 2] = action

def preprocess_state(state):
    """
    Convert state (of shape (Ymem, Xmem)) to a PyTorch tensor of shape (4,128,128).
    Here, we tile the single-channel state to create 4 channels.
    """
    from PIL import Image
    pil_img = Image.fromarray(state)
    pil_img = pil_img.resize((128, 128), Image.BILINEAR)
    state_resized = np.array(pil_img)
    stacked = np.stack([state_resized] * 4, axis=0)  # shape (4,128,128)
    return torch.from_numpy(stacked).float()

def train(num_episodes=100000):
    print("Starting training...")
    launch_dolphin()
    
    shm = wait_for_shared_memory()
    shm_array = np.ndarray(data_shape, dtype=np.float32, buffer=shm.buf)
    print("Initializing agent...")
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
        device='cuda'  # adjust as needed 
    )
    
    max_steps_per_episode = 1000
    os.makedirs("checkpoints", exist_ok=True)
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1} starting...")
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done and steps < max_steps_per_episode:
            print(f"Episode {episode+1}, Step {steps+1}: Reading shared state...")
            state_np, reward, terminal, speed, lap_progress = read_shared_state(shm_array)
            print(f"Episode {episode+1}, Step {steps+1}: Preprocessing state...")
            obs_tensor = preprocess_state(state_np)
            print(f"Episode {episode+1}, Step {steps+1}: Selecting action...")
            action = agent.select_action(obs_tensor)
            print(f"Episode {episode+1}, Step {steps+1}: Action = {action}, Reward = {reward:.3f}, Speed = {speed:.3f}, Lap Progress = {lap_progress:.3f}")
            write_action(shm_array, action)
            obs_np_preproc = obs_tensor.cpu().numpy()
            print(f"Episode {episode+1}, Step {steps+1}: Storing transition...")
            agent.store_transition((obs_np_preproc, action, reward, obs_np_preproc, terminal))
            print(f"Episode {episode+1}, Step {steps+1}: Updating agent...")
            agent.update()
            agent.frame_count += 1

            episode_reward += reward
            steps += 1

            if terminal > 0:
                print(f"Episode {episode+1}, Step {steps}: Terminal reached.")
                done = True
        
        print(f"Episode {episode+1} finished: Total Reward = {episode_reward:.3f}, Steps = {steps}")
        checkpoint_path = f"checkpoints/agent_episode_{episode+1}.pth"
        torch.save({
            'episode': episode+1,
            'frame_count': agent.frame_count,
            'state_dict': agent.online_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    print("Training finished.")

if __name__ == "__main__":
    train(num_episodes=100000)
