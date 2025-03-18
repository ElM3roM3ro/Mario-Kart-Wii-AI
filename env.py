import time
import collections
import sys
import random
sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")
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
shm_name = 'dolphin_shared'
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
    {"A": True, "B": False},   # Default remote state (A pressed)
    {"A": True, "B": True},    # Alternate remote state (for drift, etc.)
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

def compute_reward():
    """
    New reward function:
      - Rewards small increments in lap progress (every 0.1) scaled by speed.
      - Awards a bonus for crossing whole number lap progress boundaries.
      - Provides an additional bonus if speed is above 50 (none if below).
      - Marks terminal when lap progress >= 4.
    """
    global last_lap_progress
    state = read_game_state()
    if state is None:
        return 0.0, 0.0, 0.0, 0.0

    speed = state['speed']
    lap_progress = state['lap_progress']
    reward = 0.0

    # Calculate lap progress difference; only consider positive progress.
    if last_lap_progress is None:
        lap_diff = 0.0
    else:
        lap_diff = max(0.0, lap_progress - last_lap_progress)

    # Reward for small increments in lap progress:
    # Every 0.1 progress gives a base reward of 10, scaled by (speed / 50).
    progress_reward = (lap_diff / 0.1) * 10 * (speed / 50)

    # Bonus for crossing whole number boundaries in lap progress.
    whole_progress_bonus = 0
    if last_lap_progress is not None:
        last_whole = int(last_lap_progress)
        current_whole = int(lap_progress)
        if current_whole > last_whole:
            # Award 200 points for each whole number passed.
            whole_progress_bonus = (current_whole - last_whole) * 200

    # Additional bonus for speed above 50.
    speed_bonus = 0
    if speed >= 45:
        # For every point above 50, add 2 extra points.
        speed_bonus = (speed - 45) * 2

    # Sum up the reward.
    reward = progress_reward + whole_progress_bonus + speed_bonus

    # Terminal condition: if lap progress reaches 4, grant a huge bonus and signal termination.
    terminal = 0.0
    if lap_progress >= 4:
        reward += 1000  # Big reward for completing the lap.
        terminal = 1.0
    
    # --- New drifting penalty ---
    # Assume drifting actions are indices 3, 4, 5, 6, 7, and 8.
    drifting_actions = {3, 4, 5, 6, 7, 8}
    if last_action in drifting_actions and speed < 45:
        penalty = 10  # Negative penalty for drifting at low speed.
        reward -= penalty

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

def on_framedrawn(width: int, height: int, data_bytes: bytes):
    global timestep, fps_counter, last_fps_time, resetting, low_speed_counter
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
    print(f"env.py: t={timestep}, Reward={reward}, Terminal={terminal}, Speed={speed}, LapProgress={lap_progress}")
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
    if speed < 50:
        low_speed_counter += 1
    else:
        low_speed_counter = 0

    if low_speed_counter >= 720 and not resetting:
        print("Low speed detected for 720 consecutive frames. Initiating environment reset...")
        low_speed_counter = 0  # Reset the counter after triggering the reset.
        resetting = True
        reset_environment(initial=False)
        return

    # Trigger reset if terminal condition is met and we're not already resetting.
    if terminal == 1.0 and not resetting:
        print("Terminal condition reached. Initiating environment reset...")
        resetting = True
        reset_environment(initial=False)

event.on_framedrawn(on_framedrawn)

def apply_action(action_index):
    global current_wiimote_state, timestep

    last_action = action_index

    if action_index == 12:
        # For action 12, after timestep 300, hold A while swinging.
        if timestep > 300:
            new_wiimote_state = {"A": True, "B": False}
        else:
            new_wiimote_state = {"A": False, "B": False}
        current_wiimote_state = new_wiimote_state.copy()
        controller.set_wiimote_buttons(0, current_wiimote_state)
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    else:
        # For all other actions, we want A to be held.
        if action_index == 11:
            # Action 11: Combined wheelie + vertical swing.
            new_wiimote_state = {"A": True, "B": False}
            controller.set_wiimote_buttons(0, new_wiimote_state)
            controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
            nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
            controller.set_wii_nunchuk_buttons(0, nunchuck_action)
            current_wiimote_state = new_wiimote_state.copy()
            return
        else:
            # For all other actions, force A to be True.
            if 4 <= action_index <= 9:
                new_wiimote_state = wiimote_actions[1].copy()  # A is True, B is True.
            else:
                new_wiimote_state = wiimote_actions[0].copy()  # A is True, B is False.
            current_wiimote_state = new_wiimote_state.copy()
            controller.set_wiimote_buttons(0, current_wiimote_state)

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
        savestate.load_from_file(r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\funky_flame_delfino_savestate.sav")
        # Optionally, choose a random savestate slot or wait as needed.
        # time.sleep(3)

    # Wait until Dolphin updates a specific shared memory value.
    while True:
        if shm_array[0, 1] == timestep:
            break
        time.sleep(0.01)

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
