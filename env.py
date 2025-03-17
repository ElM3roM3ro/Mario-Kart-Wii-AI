import time
import collections
import sys
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

# Parameters for shared memory
Ymem = 78
Xmem = 94  
# Shared memory format (row 0):
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

target_width = 128
target_height = 128
frame_buffer = collections.deque(maxlen=4)

# --- Translated Actions for Wii Controllers ---
# Wii Remote actions: only two states are needed.
wiimote_actions = [
    {"A": True, "B": False},
    {"A": True, "B": True},
]

# Nunchuck actions: steering is based solely on StickX.
# StickY is neutral (0.0) when going straight.
nunchuck_actions = [
    {"StickX": 0.0, "StickY": 0.0, "Z": False},   # Index 1: Straight.
    {"StickX": -0.6, "StickY": 0.0, "Z": False},   # Index 2: Steer left.
    {"StickX": 0.6, "StickY": 0.0, "Z": False},    # Index 3: Steer right.
    {"StickX": -0.3, "StickY": 0.0, "Z": False},   # Index 4: Slight left (drift).
    {"StickX": 0.3, "StickY": 0.0, "Z": False},    # Index 5: Slight right (drift).
    {"StickX": -0.6, "StickY": 0.0, "Z": False},   # Index 6: More left (drift).
    {"StickX": 0.6, "StickY": 0.0, "Z": False},    # Index 7: More right (drift).
    {"StickX": -1.0, "StickY": 0.0, "Z": False},   # Index 8: Full left (drift).
    {"StickX": 1.0, "StickY": 0.0, "Z": False},    # Index 9: Full right (drift).
    {"StickX": 0.0, "StickY": 0.0, "Z": False},    # Index 10: Neutral (for Up).
    {"StickX": 0.0, "StickY": 0.0, "Z": True},     # Index 11: Neutral with item usage.
]

current_action = 0  
timestep = 0        

last_position = None
last_lap_progress = None
low_speed_counter = 0  

POS_BASE = 0x809C2EF8
OFF_X = 0x40
OFF_Y = 0x44
OFF_Z = 0x48

LAP_PROGRESS_ADDR = 0x809BD730 + 0xF8  
CURRENT_LAP_ADDR = 0x809BD730 + 0x111  
MAX_LAP_ADDR = 0x809BD730 + 0x112      

def process_frame(width, height, data_bytes):
    try:
        image = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    except Exception as e:
        print("env.py: Error in Image.frombytes:", e)
        return None
    image = image.convert("L").resize((target_width, target_height), Image.BILINEAR)
    frame = np.array(image)
    return frame

def on_framedrawn(width: int, height: int, data_bytes: bytes):
    frame = process_frame(width, height, data_bytes)
    if frame is None:
        print("env.py: Frame processing failed.")
        return
    frame_buffer.append(frame)
    if len(frame_buffer) < 4:
        return
    state_img = np.stack(list(frame_buffer), axis=0)
    reward, terminal, speed, lap_progress = compute_reward_debug()
    global timestep
    timestep += 1
    print(f"env.py: t={timestep}, Reward={reward}, Terminal={terminal}, Speed={speed}, LapProgress={lap_progress}")
    shm_array[0, 0] = timestep
    shm_array[0, 1] = timestep
    # Store the current action index.
    shm_array[0, 2] = current_action
    shm_array[0, 3] = reward
    shm_array[0, 4] = terminal
    shm_array[0, 5] = speed
    shm_array[0, 6] = lap_progress
    pil_state = Image.fromarray(state_img[0])
    pil_state = pil_state.resize((Xmem, Ymem), Image.BILINEAR)
    state_down = np.array(pil_state)
    shm_array[1:, :] = state_down

event.on_framedrawn(on_framedrawn)

def read_game_state():
    try:
        pos_x = memory.read_f32(POS_BASE + OFF_X)
        pos_y = memory.read_f32(POS_BASE + OFF_Y)
        pos_z = memory.read_f32(POS_BASE + OFF_Z)
        lap_progress = memory.read_f32(LAP_PROGRESS_ADDR)
        current_lap = int(memory.read_f32(CURRENT_LAP_ADDR))
        max_lap = int(memory.read_f32(MAX_LAP_ADDR))
        return {
            'position': (pos_x, pos_y, pos_z),
            'lap_progress': lap_progress,
            'current_lap': current_lap,
            'max_lap': max_lap
        }
    except Exception as e:
        print("env.py: Error reading game state:", e)
        return None

def compute_reward_debug():
    global last_position, last_lap_progress, low_speed_counter
    state = read_game_state()
    if state is None:
        return 0.0, 0.0, 0.0, 0.0
    current_position = state['position']
    current_lap_progress = state['lap_progress']
    current_lap = state['current_lap']
    max_lap = state['max_lap']
    speed = 0.0
    if last_position is not None:
        dx = current_position[0] - last_position[0]
        dy = current_position[1] - last_position[1]
        dz = current_position[2] - last_position[2]
        speed = (dx**2 + dy**2 + dz**2) ** 0.5
    progress_reward = 0.0
    if last_lap_progress is not None:
        if current_lap > 0 and current_lap > int(memory.read_f32(CURRENT_LAP_ADDR)):
            progress_reward = (1.0 - last_lap_progress + current_lap_progress) * 100.0
        else:
            progress_reward = max(0.0, (current_lap_progress - last_lap_progress)) * 100.0
    last_position = current_position
    last_lap_progress = current_lap_progress
    SPEED_THRESHOLD = 0.05
    if speed < SPEED_THRESHOLD:
        low_speed_counter += 1
    else:
        low_speed_counter = 0
    low_speed_penalty = -10.0 if low_speed_counter >= 10 else 0.0
    race_complete_bonus = 0.0
    terminal = 0.0
    if current_lap >= max_lap and current_lap_progress > 0.99:
        race_complete_bonus = 10.0
        terminal = 1.0
    total_reward = progress_reward + low_speed_penalty + race_complete_bonus
    return float(total_reward), terminal, speed, current_lap_progress

def apply_action(action_index):
    # For the Wii remote, if the action index is between 4 and 9, use the drift state.
    if 4 <= action_index <= 9:
        remote_action = wiimote_actions[1]
    else:
        remote_action = wiimote_actions[0]
    
    # For the nunchuck, use the corresponding action from the nunchuck_actions list if available.
    if 0 <= action_index < len(nunchuck_actions):
        nunchuck_action = nunchuck_actions[action_index]
    else:
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
    
    controller.set_wiimote_buttons(0, remote_action)
    controller.set_wii_nunchuk_buttons(0, nunchuck_action)

import threading
def main_loop():
    while True:
        time.sleep(0.03)
        apply_action(current_action)

if __name__ == "__main__":
    threading.Thread(target=main_loop, daemon=True).start()
