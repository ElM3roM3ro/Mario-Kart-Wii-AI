import time
import collections
import sys
import random
import os
import numpy as np
from PIL import Image
import torch
from multiprocessing import shared_memory

# Set user-specific site-packages paths.
user = "Zach"
if user == "Nolan":
    sys.path.append(r"C:\Users\nolan\AppData\Local\Programs\Python\Python312\Lib\site-packages")
elif user == "Zach":
    sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")
elif user == "Victor":
    sys.path.append(r"C:\Users\victo\AppData\Local\Programs\Python\Python312\Lib\site-packages")

# Import Dolphin modules.
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

# --- Parameters ---
# Shared memory: originally Ymem and Xmem were for a single downsampled frame.
# Now, we want to store 4 frames. We set:
Ymem = 78           # downsampled height of each frame
Xmem = 94           # downsampled width of each frame
num_frames = 4      # number of frames to stack

# Total shared memory shape: row 0 reserved for metadata; rows 1 to (num_frames*Ymem) hold image data.
SHM_ROWS = 1 + num_frames * Ymem  # e.g. 1 + 4*78 = 313
SHM_COLS = Xmem
shm_name = os.environ.get("SHM_NAME", "dolphin_shared")

# Create (or attach to) shared memory.
data = np.zeros((SHM_ROWS, SHM_COLS), dtype=np.float32)
try:
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=shm_name)
    print("env_multi.py: Created new shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(create=False, name=shm_name)
    print("env_multi.py: Attached to existing shared memory.")
shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shm_array[:] = data[:]

# Image processing parameters.
target_width = 128
target_height = 128
frame_buffer = collections.deque(maxlen=num_frames)

# --- Controller Actions ---
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
# Note: indices 11 and 12 are additional actions that your agent's action space should support.

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
DRIVE_DIR_ADDR = 0x802c360f  # 1:Forward, 2:Backward
MINUTES_ADDR = 0x80e48df9
SECONDS_ADDR = 0x80e48dfa
MILLISECONDS_ADDR = 0x80e48dfc
SPEED_ADDR = 0x80fad2c4       # or 0x80fad2c8, adjust as needed
LAP_PROGRESS_ADDR = 0x80e48d3c  # or 0x80e48d38, adjust as needed
CURRENT_LAP_ADDR = 0x80e96428
MAX_LAP_ADDR = 0x80e96428     

# Persistent state for the Wii remote.
current_wiimote_state = wiimote_actions[0].copy()

def process_frame(width, height, data_bytes):
    try:
        image = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    except Exception as e:
        print("env_multi.py: Error in Image.frombytes:", e)
        return None
    # Convert to grayscale and resize to target dimensions.
    image = image.convert("L").resize((target_width, target_height), Image.BILINEAR)
    return np.array(image)

# Global variables for tracking progress and time.
last_lap_progress = None
last_elapsed_time = None  # tracks previous elapsed time

# --- Configurable Parameters ---
PROGRESS_UNIT = 0.0143
SPEED_SCALE = 50.0
SPEED_THRESHOLD = 80.0
TERMINAL_BONUS = 10.0
DRIFTING_PENALTY = 1.0
PREMATURE_A_PENALTY = 1.0
DRIFT_SPEED_THRESHOLD = 45.0
NORMALIZATION_FACTOR = 1000.0
LAMBDA = 1.0
MU = 0.5

def compute_reward():
    """
    Computes reward based on speed and lap progress.
    Returns: normalized_reward, terminal, speed, lap_progress
    """
    global last_lap_progress
    state = read_game_state()
    if state is None:
        return 0.0, 0.0, 0.0, 0.0
    speed = state['speed']
    lap_progress = state['lap_progress']
    reward = speed / 2500
    if last_lap_progress is None:
        last_lap_progress = lap_progress
    lap_diff = lap_progress - last_lap_progress
    if lap_diff >= 0.0001:
        num_increments = int(lap_diff / 0.0001)
        reward += 0.0025 * num_increments
    if int(lap_progress) > int(last_lap_progress) and int(last_lap_progress) != 0:
        reward += 10
    terminal = 1.0 if lap_progress >= 4.0 else 0.0
    last_lap_progress = lap_progress
    return float(reward), terminal, speed, lap_progress

def read_game_state():
    try:
        speed = memory.read_f32(SPEED_ADDR)
        lap_progress = memory.read_f32(LAP_PROGRESS_ADDR)
        current_lap = int(memory.read_f32(CURRENT_LAP_ADDR))
        max_lap = int(memory.read_f32(MAX_LAP_ADDR))
        return {'speed': speed, 'lap_progress': lap_progress, 'current_lap': current_lap, 'max_lap': max_lap}
    except Exception as e:
        print("env_multi.py: Error reading game state:", e)
        return None

# Global variable for held action.
held_action = 0

def on_framedrawn(width: int, height: int, data_bytes: bytes):
    global timestep, fps_counter, last_fps_time, resetting, low_speed_counter, held_action
    frame = process_frame(width, height, data_bytes)
    if frame is None:
        print("env_multi.py: Frame processing failed.")
        return
    frame_buffer.append(frame)
    if len(frame_buffer) < num_frames:
        return

    # Stack 4 distinct frames (each of shape (target_height, target_width)).
    state_img = np.stack(list(frame_buffer), axis=0)  # shape: (4, 128, 128)
    reward, terminal, speed, lap_progress = compute_reward()
    timestep += 1

    # Write metadata.
    shm_array[0, 0] = timestep
    shm_array[0, 1] = timestep
    shm_array[0, 3] = reward
    shm_array[0, 4] = terminal
    shm_array[0, 5] = speed
    shm_array[0, 6] = lap_progress

    # Downsample each frame from (128, 128) to (Xmem, Ymem) using bilinear interpolation.
    stacked_down = []
    for i in range(num_frames):
        pil_img = Image.fromarray(state_img[i])
        pil_img = pil_img.resize((Xmem, Ymem), Image.BILINEAR)
        stacked_down.append(np.array(pil_img))
    stacked_down = np.stack(stacked_down, axis=0)  # shape: (4, Ymem, Xmem)
    # Flatten vertically: shape becomes (4*Ymem, Xmem)
    flat_state = stacked_down.reshape(num_frames * Ymem, Xmem)
    shm_array[1:, :] = flat_state

    # Low-speed reset condition.
    if speed < 45:
        resetting = True
        penalty = 0.5
        reward -= penalty
        shm_array[0, 3] = reward
        reset_environment(initial=False)
        return

    # Terminal reset condition.
    if terminal == 1.0 and not resetting:
        print("Terminal condition reached. Initiating environment reset...")
        resetting = True
        reset_environment(initial=False)

    # Hold actions.
    new_action = int(shm_array[0, 2])
    if new_action != held_action:
        held_action = new_action
    apply_action(held_action)

# Register the on_framedrawn callback.
event.on_framedrawn(on_framedrawn)

def apply_action(action_index):
    global timestep, last_action
    last_action = action_index
    if action_index == 13:
        controller.set_wiimote_buttons(0, {"A": False, "B": False})
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    if action_index == 12:
        controller.set_wiimote_buttons(0, {"A": False, "B": False})
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    if action_index == 11:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    if 4 <= action_index <= 9:
        controller.set_wiimote_buttons(0, {"A": True, "B": True})
    else:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
    if 0 <= action_index < len(nunchuck_actions):
        nunchuck_action = nunchuck_actions[action_index]
    else:
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
    controller.set_wii_nunchuk_buttons(0, nunchuck_action)

def reset_environment(initial=False):
    global timestep, last_lap_progress, resetting, shm_array, frame_buffer
    print("Resetting environment...")
    timestep = 0
    shm_array[0, 1] = timestep
    last_lap_progress = None
    frame_buffer.clear()
    if initial:
        print("Initial reset: Using CLI savestate.")
    else:
        reset_choice = random.randint(1, 4)
        if reset_choice == 1:
            savestate.load_from_file(r"funky_flame_delfino_savestate_startv2.sav")
        elif reset_choice == 2:
            savestate.load_from_file(r"funky_flame_delfino_savestate2.sav")
        elif reset_choice == 3:
            savestate.load_from_file(r"funky_flame_delfino_savestate3.sav")
        elif reset_choice == 4:
            savestate.load_from_file(r"funky_flame_delfino_savestate4.sav")
    while True:
        if shm_array[0, 1] == timestep:
            break
        time.sleep(0.01)
    shm_array[0, 3] = 0.0  # Reward.
    shm_array[0, 4] = 0.0  # Terminal.
    # Clear the state area.
    shm_array[1:, :] = np.zeros((SHM_ROWS-1, SHM_COLS), dtype=np.float32)
    timestep += 1
    shm_array[0, 0] = timestep
    print("Environment reset complete.")
    resetting = False

def main_loop():
    while True:
        time.sleep(0.03)
        current_action = int(shm_array[0, 2])
        apply_action(current_action)

# For initial start, perform an initial reset (Dolphin loads CLI savestate).
reset_environment(initial=True)

import threading
threading.Thread(target=main_loop, daemon=True).start()
