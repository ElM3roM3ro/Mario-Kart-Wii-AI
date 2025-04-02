import time
import collections
import sys
import random

user = "Zach"
if user == "Nolan":
    sys.path.append(r"C:\Users\nolan\AppData\Local\Programs\Python\Python312\Lib\site-packages")  # Nolan's path
elif user == "Zach":
    sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")  # Zach's path
elif user == "Victor":
    sys.path.append(r"C:\Users\victo\AppData\Local\Programs\Python\Python312\Lib\site-packages")  # Victor's path

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
Ymem = 128
Xmem = 128  
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

# --- New Action Mapping ---
#
# Define 8 actions:
# 0: Wheelie
# 1: Slight drift left
# 2: Slight drift right
# 3: More drift left
# 4: More drift right
# 5: Full drift left
# 6: Full drift right
# 7: Use item
#
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
last_elapsed_time = None  # To track time from the previous step.

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
    Simplified reward function:
      - Base reward: current speed.
      - For every 0.001 increment in lap progress, add 0.1 reward.
      - When lap progress crosses a whole number, add bonus.
      - Terminal flag is set if lap_progress >= 4.
      
    Returns:
      normalized_reward (float), terminal (float), speed (float), lap_progress (float)
    """
    global last_lap_progress
    
    state = read_game_state()  # Expects a dict with 'speed' and 'lap_progress'
    if state is None:
        return 0.0, 0.0, 0.0, 0.0

    speed = state['speed']
    lap_progress = state['lap_progress']
    reward = 0.0

    if last_lap_progress is None:
        last_lap_progress = lap_progress

    lap_diff = lap_progress - last_lap_progress
    if lap_diff >= 0.001:
        num_increments = int(lap_diff / 0.001)
        reward += 0.1 * num_increments

        if int(lap_progress) > int(last_lap_progress) and int(last_lap_progress) != 0:
            reward += 3
        
        last_lap_progress = lap_progress

    terminal = 1.0 if lap_progress >= 4.0 else 0.0

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

    state_img = np.stack(list(frame_buffer), axis=0)
    reward, terminal, speed, lap_progress = compute_reward()
    timestep += 1

    shm_array[0, 0] = timestep
    shm_array[0, 1] = timestep
    shm_array[0, 3] = reward
    shm_array[0, 4] = terminal
    shm_array[0, 5] = speed
    shm_array[0, 6] = lap_progress

    speed_reset = False
    if speed < 65:
        low_speed_counter += 1
        shm_array[0, 3] -= 0.01

    if low_speed_counter == 80:
        speed_reset = True
        
    if speed < 65 and speed_reset:
        resetting = True
        penalty = 2
        shm_array[0, 3] -= penalty
        low_speed_counter = 0
        # Set terminal flag to signal reset due to speed penalty.
        shm_array[0, 4] = 1.0
        # Call reset_environment with preserve_terminal=True so we hold the terminal flag briefly.
        reset_environment(initial=False, preserve_terminal=True)
        return

    state_down = state_img[0]
    shm_array[1:, :] = state_down

    if terminal == 1.0 and not resetting:
        #print("Terminal condition reached. Initiating environment reset...")
        resetting = True
        reset_environment(initial=False)
    
    new_action = int(shm_array[0, 2])
    apply_action(new_action)

event.on_framedrawn(on_framedrawn)

def apply_action(action_index):
    global timestep, last_action
    last_action = action_index

    if action_index == 0:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 1.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 7:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": True}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 1:
        stick_action = {"StickX": -0.3, "StickY": 0.0, "Z": False}
    elif action_index == 2:
        stick_action = {"StickX": 0.3, "StickY": 0.0, "Z": False}
    elif action_index == 3:
        stick_action = {"StickX": -0.6, "StickY": 0.0, "Z": False}
    elif action_index == 4:
        stick_action = {"StickX": 0.6, "StickY": 0.0, "Z": False}
    elif action_index == 5:
        stick_action = {"StickX": -1.0, "StickY": 0.0, "Z": False}
    elif action_index == 6:
        stick_action = {"StickX": 1.0, "StickY": 0.0, "Z": False}
    else:
        stick_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}

    controller.set_wiimote_buttons(0, {"A": True, "B": True})
    controller.set_wii_nunchuk_buttons(0, stick_action)

def reset_environment(initial=False, preserve_terminal=False):
    global timestep, last_lap_progress, resetting, shm_array, frame_buffer
    #print("Resetting environment...")
    timestep = 0
    shm_array[0, 1] = timestep
    last_lap_progress = None
    frame_buffer.clear()

    if not initial:
        reset_choice = random.randint(1, 4)
        if reset_choice == 1:
            savestate.load_from_file(r"funky_flame_delfino_savestate_startv2.sav")
        if reset_choice == 2:
            savestate.load_from_file(r"funky_flame_delfino_savestate2.sav")
        if reset_choice == 3:
            savestate.load_from_file(r"funky_flame_delfino_savestate3.sav")
        if reset_choice == 4:
            savestate.load_from_file(r"funky_flame_delfino_savestate4.sav")

    while True:
        if shm_array[0, 1] == timestep:
            break

    shm_array[0, 3] = 0.0
    #if preserve_terminal:
        #time.sleep(0.0167)  # Brief pause to allow the terminal flag to be detected by the trainer.
    # Always clear the terminal flag after reset.
    shm_array[0, 4] = 0.0
    shm_array[1:, :] = np.zeros((Ymem, Xmem), dtype=np.float32)
    
    timestep += 1
    shm_array[0, 0] = timestep

    #print("Environment reset complete.")
    resetting = False

reset_environment(initial=True)

while True:
    await event.framedrawn()
    current_action = int(shm_array[0, 2])
    apply_action(current_action)
