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
# Instead of storing a full stack of frames, we now store one frame.
Ymem = 78           # downsampled height of each frame
Xmem = 94           # downsampled width of each frame
SHM_ROWS = 1 + Ymem  # row 0 for metadata; rows 1..Ymem hold a single downsampled frame
SHM_COLS = Xmem
shm_name = os.environ.get("SHM_NAME", "dolphin_shared")
data = np.zeros((SHM_ROWS, SHM_COLS), dtype=np.float32)
try:
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=shm_name)
    print("vec_env.py: Created new shared memory.")
except FileExistsError:
    shm = shared_memory.SharedMemory(create=False, name=shm_name)
    print("vec_env.py: Attached to existing shared memory.")
shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
shm_array[:] = data[:]

# Image processing parameters.
target_width = 128
target_height = 128

def process_frame(width, height, data_bytes):
    try:
        image = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    except Exception as e:
        print("vec_env.py: Error in Image.frombytes:", e)
        return None
    # Convert to grayscale and resize to target dimensions.
    image = image.convert("L").resize((target_width, target_height), Image.BILINEAR)
    return np.array(image)

# Global variables for tracking progress and game state.
timestep = 0        
last_lap_progress = None
low_speed_counter = 0
last_action = 0
fps_counter = 0
last_fps_time = time.time()
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
wiimote_actions = [
    {"B": False},
    {"B": True},
]
nunchuck_actions = [
        {"StickX": 0.0, "StickY": 0.0, "Z": False},
        {"StickX": -0.6, "StickY": 0.0, "Z": False},
        {"StickX": 0.6, "StickY": 0.0, "Z": False},
        {"StickX": -0.3, "StickY": 0.0, "Z": False},
        {"StickX": 0.3, "StickY": 0.0, "Z": False},
        {"StickX": -0.6, "StickY": 0.0, "Z": False},
        {"StickX": 0.6, "StickY": 0.0, "Z": False},
        {"StickX": -1.0, "StickY": 0.0, "Z": False},
        {"StickX": 1.0, "StickY": 0.0, "Z": False},
        {"StickX": 0.0, "StickY": 0.0, "Z": False},
        {"StickX": 0.0, "StickY": 0.0, "Z": True},
    ]
current_wiimote_state = wiimote_actions[0].copy()

def compute_reward():
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
        print("vec_env.py: Error reading game state:", e)
        return None

def on_framedrawn(width: int, height: int, data_bytes: bytes):
    global timestep, fps_counter, last_fps_time, resetting, low_speed_counter, last_action
    frame = process_frame(width, height, data_bytes)
    if frame is None:
        print("vec_env.py: Frame processing failed.")
        return
    # Downsample the processed frame from (128,128) to (Xmem, Ymem)
    pil_img = Image.fromarray(frame)
    pil_img = pil_img.resize((Xmem, Ymem), Image.BILINEAR)
    downsampled = np.array(pil_img)
    
    fps_counter += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        print(f"VEC_ENV FPS: {fps_counter}")
        fps_counter = 0
        last_fps_time = current_time

    reward, terminal, speed, lap_progress = compute_reward()
    timestep += 1

    # Write metadata.
    shm_array[0, 0] = timestep
    shm_array[0, 1] = timestep
    shm_array[0, 3] = reward
    shm_array[0, 4] = terminal
    shm_array[0, 5] = speed
    shm_array[0, 6] = lap_progress

    # Write the current downsampled frame (shape: (Ymem, Xmem)) to shared memory rows 1:.
    shm_array[1:, :] = downsampled

    if speed < 45:
        resetting = True
        penalty = 0.5
        reward -= penalty
        shm_array[0, 3] = reward
        reset_environment(initial=False)
        return

    if terminal == 1.0 and not resetting:
        print("Terminal condition reached. Initiating environment reset...")
        resetting = True
        reset_environment(initial=False)

    new_action = int(shm_array[0, 2])
    if new_action != last_action:
        last_action = new_action
    apply_action(last_action)

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
    if action_index < len(nunchuck_actions):
        nunchuck_action = nunchuck_actions[action_index]
    else:
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
    controller.set_wii_nunchuk_buttons(0, nunchuck_action)

def reset_environment(initial=False):
    global timestep, last_lap_progress, resetting, shm_array
    print("Resetting environment...")
    timestep = 0
    shm_array[0, 1] = timestep
    last_lap_progress = None
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
    shm_array[0, 3] = 0.0
    shm_array[0, 4] = 0.0
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

import threading
threading.Thread(target=main_loop, daemon=True).start()
