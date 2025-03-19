import time
import collections
import sys
import random
user = "Zach"
if user == "Nolan":
    sys.path.append(r"C:\Users\nolan\AppData\Local\Programs\Python\Python312\Lib\site-packages") # Nolan's path
elif user == "Zach":
    sys.path.append(r"F:\Python\3.12.0\Lib\site-packages") # Zach's path
import numpy as np
from PIL import Image
import torch
from multiprocessing import shared_memory

try:
    from dolphin import event, controller, memory, savestate
except ImportError:
    print("Dolphin modules not found. Using dummy objects for development.")
    class Dummy:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    event = Dummy()
    controller = Dummy()
    memory = Dummy()
    savestate = Dummy()

# Shared memory parameters.
Ymem = 78
Xmem = 94  
# Shared memory layout (row 0):
# [Dolphin timestep, Emulator timestep, action, reward, terminal, speed, lap_progress]
# Rows 1: state (downsampled image data)
import os
shm_name = os.environ.get("SHM_NAME", "dolphin_shared")

data = np.zeros((Ymem + 1, Xmem), dtype=np.float32)
try:
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=shm_name)
    print("env.py: Created new shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(create=False, name=shm_name)
    print("env.py: Attached to existing shared memory.")
shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shm_array[:] = data[:]

# Image processing parameters.
target_width = 128
target_height = 128
frame_buffer = collections.deque(maxlen=4)

# --- Translated Actions for Wii Controllers ---
wiimote_actions = [
    {"B": False},   # Default remote state (A pressed)
    {"B": True},    # Alternate remote state (for drift, etc.)
]
nunchuck_actions = [
    {"StickX": 0.0, "StickY": 0.0, "Z": False},   # Index 0: Straight.
    {"StickX": -0.6, "StickY": 0.0, "Z": False},   # Index 1: Steer left.
    {"StickX": 0.6, "StickY": 0.0, "Z": False},    # Index 2: Steer right.
    {"StickX": -0.3, "StickY": 0.0, "Z": False},   # Index 3: Slight left (drift).
    {"StickX": 0.3, "StickY": 0.0, "Z": False},    # Index 4: Slight right (drift).
    {"StickX": -0.6, "StickY": 0.0, "Z": False},   # Index 5: More left (drift).
    {"StickX": 0.6, "StickY": 0.0, "Z": False},    # Index 6: More right (drift).
    {"StickX": -1.0, "StickY": 0.0, "Z": False},   # Index 7: Full left (drift).
    {"StickX": 1.0, "StickY": 0.0, "Z": False},    # Index 8: Full right (drift).
    {"StickX": 0.0, "StickY": 0.0, "Z": False},    # Index 9: Neutral.
    {"StickX": 0.0, "StickY": 0.0, "Z": True},     # Index 10: Neutral with item usage.
]
# Note: Your agent’s action space should now include these two new actions (indices 11 and 12).

current_action = 0  
timestep = 0        
last_lap_progress = None
low_speed_counter = 0
last_action = 0

# Globals for FPS calculation.
fps_counter = 0
last_fps_time = time.time()

# Flag to prevent overlapping resets.
resetting = False

# Memory addresses for game state.
MINUTES_ADDR = 0x80e48df9
SECONDS_ADDR = 0x80e48dfa
MILLISECONDS_ADDR = 0x80e48dfc
SPEED_ADDR = 0x80fad2c4       # or 0x80fad2c8, adjust as needed
LAP_PROGRESS_ADDR = 0x80e48d3c # or 0x80e48d38, adjust as needed
CURRENT_LAP_ADDR = 0x80e96428
MAX_LAP_ADDR = 0x80e96428     

# Persistent state for the Wii remote to avoid repeated presses.
current_wiimote_state = wiimote_actions[0].copy()

def process_frame(width, height, data_bytes):
    try:
        image = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    except Exception as e:
        print("env.py: Error in Image.frombytes:", e)
        return None
    image = image.convert("L").resize((target_width, target_height), Image.BILINEAR)
    return np.array(image)

# Global variables for tracking progress and time.
last_lap_progress = None
last_elapsed_time = None  # New global to track time from the previous step.

def read_time():
    """
    Reads the in-game timer values (minutes, seconds, and milliseconds) and computes
    the elapsed time in seconds.
    Note: ms is multiplied by 100 per your comment; adjust if needed.
    """
    try:
        minutes = memory.read_f32(MINUTES_ADDR)
        seconds = memory.read_f32(SECONDS_ADDR)
        ms = memory.read_f32(MILLISECONDS_ADDR) * 100  # Adjust if necessary.
        # Convert ms to seconds.
        elapsed_time = minutes * 60 + seconds + ms / 1000.0
        return elapsed_time
    except Exception as e:
        print("Error reading time:", e)
        return 0.0

# --- Configurable Parameters ---
PROGRESS_UNIT = 0.0143          # Reward unit per 0.1 progress increment.
SPEED_SCALE = 50.0             # Scale factor for speed in progress reward.
SPEED_THRESHOLD = 80.0         # Threshold for bonus speed reward.
TERMINAL_BONUS = 10.0          # Bonus reward when lap_progress reaches the terminal condition.
DRIFTING_PENALTY = 1.0         # Penalty for drifting at low speed.
PREMATURE_A_PENALTY = 1.0      # Penalty for holding A too early
DRIFT_SPEED_THRESHOLD = 45.0   # Speed below which drifting is penalized.
NORMALIZATION_FACTOR = 1000.0   # Adjust as needed (was commented as 800, so confirm the intended scale).
LAMBDA = 1.0                 # Scaling factor for lap progress in potential.
MU = 0.5                     # Scaling factor for elapsed time in potential.

# --- Modular Reward Components ---

def compute_progress_reward(lap_progress, last_lap_progress, speed):
    """Calculate the reward based on lap progress increments, bonus for whole numbers, and speed bonus."""
    if last_lap_progress is None:
        return 0.0
    lap_diff = max(0.0, lap_progress - last_lap_progress)
    progress_reward = (lap_diff / PROGRESS_UNIT) * (speed / SPEED_SCALE)
    
    # Bonus for crossing whole number boundaries.
    whole_bonus = 0
    if last_lap_progress is not None:
        last_whole = int(last_lap_progress)
        current_whole = int(lap_progress)
        if current_whole > last_whole:
            if last_whole >= 1:
                whole_bonus = current_whole - last_whole
            
    # Additional bonus for speed above threshold.
    speed_bonus = (speed - SPEED_THRESHOLD) if speed >= SPEED_THRESHOLD else 0
    return progress_reward + whole_bonus + speed_bonus

def compute_shaping_reward(lap_progress, elapsed_time, last_lap_progress, last_elapsed_time):
    """Compute the potential-based shaping reward."""
    phi_new = LAMBDA * lap_progress - MU * elapsed_time
    if last_lap_progress is None or last_elapsed_time is None:
        phi_old = 0.0
    else:
        phi_old = LAMBDA * last_lap_progress - MU * last_elapsed_time
    return phi_new - phi_old

# def compute_reward():
#     """
#     Computes a normalized reward including:
#       - Progress-based reward.
#       - Potential-based shaping.
#       - Terminal bonus and drifting penalty.
#     """
#     global last_lap_progress, last_elapsed_time, last_action

#     state = read_game_state()
#     if state is None:
#         return 0.0, 0.0, 0.0, 0.0

#     speed = state['speed']
#     lap_progress = state['lap_progress']
#     elapsed_time = read_time()

#     # --- Progress Reward ---
#     base_reward = compute_progress_reward(lap_progress, last_lap_progress, speed)
    
#     # --- Potential-Based Shaping ---
#     shaping_reward = compute_shaping_reward(lap_progress, elapsed_time, last_lap_progress, last_elapsed_time)
    
#     raw_reward = base_reward + shaping_reward

#     # --- Terminal Bonus ---
#     terminal = 0.0
#     if lap_progress >= 4:
#         raw_reward += TERMINAL_BONUS
#         terminal = 1.0

#     # --- Drifting Penalty ---
#     drifting_actions = {3, 4, 5, 6, 7, 8}
#     if last_action in drifting_actions and speed < DRIFT_SPEED_THRESHOLD:
#         raw_reward -= DRIFTING_PENALTY

#     if controller.get_wiimote_buttons(0)["A"] == True and timestep < 145:
#         raw_reward -= PREMATURE_A_PENALTY

#     # Update last progress and time for the next frame.
#     last_lap_progress = lap_progress
#     last_elapsed_time = elapsed_time

#     normalized_reward = raw_reward / NORMALIZATION_FACTOR

#     return float(normalized_reward), terminal, speed, lap_progress

# Global variable to track the previous lap progress.
last_lap_progress = None

def compute_reward():
    """
    Simplified reward function:
      - Base reward: the current speed.
      - For every 0.001 increment in lap progress, add 0.01 reward.
      - When lap progress crosses a whole number, add a bonus of 10.
      - Terminal flag is set if lap_progress reaches or exceeds 4.
      
    Returns:
      normalized_reward (float), terminal (float), speed (float), lap_progress (float)
    """
    global last_lap_progress
    
    state = read_game_state()  # Assumes a function returning a dict with 'speed' and 'lap_progress'
    if state is None:
        return 0.0, 0.0, 0.0, 0.0

    speed = state['speed']
    lap_progress = state['lap_progress']
    
    # Base reward from speed.
    reward = speed / 1000

    # If this is the first call, just set last_lap_progress.
    if last_lap_progress is None:
        last_lap_progress = lap_progress

    # Compute how much lap progress increased since the last frame.
    lap_diff = lap_progress - last_lap_progress

    # Add 0.1 reward for each 0.0143 increment.
    if lap_diff >= 0.001:
        num_increments = int(lap_diff / 0.001)
        reward += 0.01 * num_increments

    # If lap progress crosses a whole number boundary, add a bonus of 10.
    if int(lap_progress) > int(last_lap_progress) and int(last_lap_progress) != 0:
        reward += 10

    # Set terminal flag if lap_progress reaches or exceeds 4 (you can adjust this value).
    terminal = 1.0 if lap_progress >= 4.0 else 0.0

    # Update last_lap_progress for the next call.
    last_lap_progress = lap_progress

    return float(reward), terminal, speed, lap_progress

def read_game_state():
    try:
        speed = memory.read_f32(SPEED_ADDR)
        lap_progress = memory.read_f32(LAP_PROGRESS_ADDR)
        current_lap = int(memory.read_f32(CURRENT_LAP_ADDR))
        max_lap = int(memory.read_f32(MAX_LAP_ADDR))
        return {
            'speed': speed,
            'lap_progress': lap_progress,
            'current_lap': current_lap,
            'max_lap': max_lap
        }
    except Exception as e:
        print("env.py: Error reading game state:", e)
        return None

# Global variable to hold the last selected action.
held_action = 0

def on_framedrawn(width: int, height: int, data_bytes: bytes):
    global timestep, fps_counter, last_fps_time, resetting, low_speed_counter, held_action
    frame = process_frame(width, height, data_bytes)
    if frame is None:
        print("env.py: Frame processing failed.")
        return
    frame_buffer.append(frame)
    if len(frame_buffer) < 4:
        return

    # Calculate and print FPS every second.
    fps_counter += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        print(f"FPS: {fps_counter}")
        fps_counter = 0
        last_fps_time = current_time

    state_img = np.stack(list(frame_buffer), axis=0)
    reward, terminal, speed, lap_progress = compute_reward()
    timestep += 1
    #print(f"env.py: t={timestep}, Reward={reward}, Terminal={terminal}, Speed={speed}, LapProgress={lap_progress}")
    shm_array[0, 0] = timestep
    shm_array[0, 1] = timestep
    shm_array[0, 3] = reward
    shm_array[0, 4] = terminal
    shm_array[0, 5] = speed
    shm_array[0, 6] = lap_progress

    pil_state = Image.fromarray(state_img[0])
    pil_state = pil_state.resize((Xmem, Ymem), Image.BILINEAR)
    state_down = np.array(pil_state)
    shm_array[1:, :] = state_down

    # New low-speed reset condition:
    if speed < 25:
        resetting = True
        penalty = 5
        reward -= penalty
        shm_array[0, 3] = reward
        reset_environment(initial=False)
        return

    # Trigger reset if terminal condition is met and we're not already resetting.
    if terminal == 1.0 and not resetting:
        print("Terminal condition reached. Initiating environment reset...")
        resetting = True
        reset_environment(initial=False)
    
    # --- Hold actions across frames ---
    # Read the new action from shared memory.
    new_action = int(shm_array[0, 2])
    # Only update held_action if a new action is provided.
    if new_action != held_action:
        held_action = new_action
        #print(f"New action selected: {held_action}")
    # Apply the held action every frame.
    apply_action(held_action)

event.on_framedrawn(on_framedrawn)

def apply_action(action_index):
    global timestep, last_action
    last_action = action_index

    if action_index == 13:
        # Do nothing
        controller.set_wiimote_buttons(0, {"A": False, "B": False})
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    # --- Special case for action 12 ---
    if action_index == 12:
        # Temporarily disable A while performing the swing.
        # (We also update B if needed; here we use B: False.)
        controller.set_wiimote_buttons(0, {"A": False, "B": False})
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return

    # --- Special case for action 11 (combined wheelie + vertical swing) ---
    if action_index == 11:
        # Do the swing command while leaving A enabled (A remains True from reset).
        controller.set_wiimote_buttons(0, {"A":True, "B": False})
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return

    # --- For all other actions, only update B (A remains True) ---
    # For drifting actions (indices 4 through 9), we want B set to True.
    if 4 <= action_index <= 9:
        controller.set_wiimote_buttons(0, {"A":True, "B": True})
    else:
        controller.set_wiimote_buttons(0, {"A":True, "B": False})

    # Update the nunchuck based on the action.
    if 0 <= action_index < len(nunchuck_actions):
        nunchuck_action = nunchuck_actions[action_index]
    else:
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
    controller.set_wii_nunchuk_buttons(0, nunchuck_action)

def reset_environment(initial=False):
    """
    Resets the environment.
    If initial==True, the reset does not load a savestate because Dolphin already loaded the CLI savestate.
    For subsequent resets (initial==False), select a savestate slot at random.
    """
    global timestep, last_lap_progress, resetting, shm_array, frame_buffer

    print("Resetting environment...")

    # Reset internal counters.
    timestep = 0
    shm_array[0, 1] = timestep
    last_lap_progress = None
    frame_buffer.clear()

    if initial:
        print("Initial reset: Using CLI savestate.")
    else:
        reset_choice = random.randint(0, 4)
        if reset_choice == 1:
            savestate.load_from_file(r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestateGoodStart.sav")
        if reset_choice == 2:
            savestate.load_from_file(r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate2.sav")
        if reset_choice == 3:
            savestate.load_from_file(r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate3.sav")
        if reset_choice == 4:
            savestate.load_from_file(r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate4.sav")
        # Optionally, choose a random savestate slot or wait as needed.
        # time.sleep(3)

    # Wait until Dolphin updates a specific shared memory value.
    while True:
        if shm_array[0, 1] == timestep:
            break
        #time.sleep(0.01)

    # Reset reward and terminal values.
    shm_array[0, 3] = 0.0  # Reward.
    shm_array[0, 4] = 0.0  # Terminal.
    # Clear the state image area.
    shm_array[1:, :] = np.zeros((Ymem, Xmem), dtype=np.float32)
    
    # Update timestep.
    timestep += 1
    shm_array[0, 0] = timestep

    print("Environment reset complete.")
    # Allow further resets.
    resetting = False

def main_loop():
    while True:
        time.sleep(0.03)
        current_action = int(shm_array[0, 2])
        apply_action(current_action)

# For initial start, let Dolphin load the savestate via CLI.
# Thus, we simply perform an initial reset without reloading a savestate slot.
reset_environment(initial=True)

import threading
threading.Thread(target=main_loop, daemon=True).start()
