import time
import collections
import sys
# Set up site–packages path.
user = "Nolan"
if user == "Nolan":
    sys.path.append(r"C:\Users\nolan\AppData\Local\Programs\Python\Python312\Lib\site-packages")
elif user == "Zach":
    sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")
elif user == "Victor":
    sys.path.append(r"C:\Users\victo\AppData\Local\Programs\Python\Python312\Lib\site-packages")
import random
import os
import numpy as np
from PIL import Image
import torch
import threading
from multiprocessing.connection import Listener
from multiprocessing import Lock

# Attempt to import Dolphin modules; use dummies for development.
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

# --- Global State and Lock ---
state_lock = Lock()
state = {
    "timestep": 0,
    "frame": np.zeros((128, 128), dtype=np.uint8),
    "reward": 0.0,
    "terminal": False,
    "speed": 0.0,
    "lap_progress": 0.0,
    "action": 0  # default action
}

# --- Server Thread ---
def server_thread():
    port = int(os.environ.get("ENV_PORT", "6000"))
    address = ('localhost', port)
    authkey = b'secret'
    listener = Listener(address, authkey=authkey)
    print(f"env_multi.py: Listening on {address} for client connections.")
    while True:
        conn = listener.accept()
        try:
            request = conn.recv()  # Expected to be a dict with "command"
            if isinstance(request, dict) and "command" in request:
                cmd = request["command"]
                if cmd == "get_state":
                    with state_lock:
                        # Return a copy of the state
                        conn.send(state.copy())
                elif cmd == "set_action":
                    new_action = request.get("value", 0)
                    with state_lock:
                        state["action"] = int(new_action)
                    conn.send({"status": "ok"})
                else:
                    conn.send({"error": "Unknown command"})
            else:
                conn.send({"error": "Invalid request format"})
        except Exception as e:
            print("env_multi.py: Error handling connection:", e)
        finally:
            conn.close()

# Start the server thread (runs in the background)
server = threading.Thread(target=server_thread, daemon=True)
server.start()

# --- Image Processing and Global Variables ---
target_width = 128
target_height = 128
frame_buffer = collections.deque(maxlen=4)
timestep = 0        
last_action = 0

# Game memory addresses.
DRIVE_DIR_ADDR = 0x802c360f  
MINUTES_ADDR = 0x80e48df9
SECONDS_ADDR = 0x80e48dfa
MILLISECONDS_ADDR = 0x80e48dfc
SPEED_ADDR = 0x80fad2c4       
LAP_PROGRESS_ADDR = 0x80e48d3c  
CURRENT_LAP_ADDR = 0x80e96428
MAX_LAP_ADDR = 0x80e96428     

def process_frame(width, height, data_bytes):
    try:
        image = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    except Exception as e:
        print("env_multi.py: Error in Image.frombytes:", e)
        return None
    image = image.convert("L").resize((target_width, target_height), Image.BILINEAR)
    return np.array(image)

last_lap_progress = None

# --- Reward and Game State Functions ---
def compute_reward():
    global last_lap_progress
    state_info = read_game_state()
    if state_info is None:
        return 0.0, 0.0, 0.0, 0.0
    speed = state_info['speed']
    lap_progress = state_info['lap_progress']
    reward = 0.0
    terminal = False
    if last_lap_progress is None:
        last_lap_progress = lap_progress
    lap_diff = lap_progress - last_lap_progress
    if lap_diff >= 0.001:
        num_increments = int(lap_diff / 0.001)
        reward += 0.1 * num_increments
        if int(lap_progress) > int(last_lap_progress) and int(last_lap_progress) != 0:
            reward += 3.0
        last_lap_progress = lap_progress
    if speed < 45:
        reward -= 10.0
        terminal = True
        print("env_multi.py: Speed too low, terminating.")
    if lap_progress >= 4.0:
        reward += 10.0
        terminal = True
        print("env_multi.py: Lap complete, terminating.")
    print(f"env_multi.py: Speed: {speed}, Lap Progress: {lap_progress}, Reward: {reward}, Terminal: {terminal}")
    return reward, terminal, speed, lap_progress

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
        print("env_multi.py: Error reading game state:", e)
        return None

def apply_action(action_index):
    global last_action
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

def reset_environment(initial=False):
    global timestep, last_lap_progress, frame_buffer, state
    timestep = 0
    with state_lock:
        state["timestep"] = timestep
    last_lap_progress = None
    frame_buffer.clear()
    if not initial:
        reset_choice = random.randint(1, 4)
        if reset_choice == 1:
            savestate.load_from_file(r"funky_flame_delfino_savestate_startv2.sav")
        elif reset_choice == 2:
            savestate.load_from_file(r"funky_flame_delfino_savestate2.sav")
        elif reset_choice == 3:
            savestate.load_from_file(r"funky_flame_delfino_savestate3.sav")
        elif reset_choice == 4:
            savestate.load_from_file(r"funky_flame_delfino_savestate4.sav")
    # Simulate a brief pause until ready
    #time.sleep(0.01)
    with state_lock:
        state["reward"] = 0.0
        state["terminal"] = False
    timestep += 1
    with state_lock:
        state["timestep"] = timestep

# Reset at startup.
reset_environment(initial=True)

# --- Frame Skip Logic ---
frame_skip = 4
frame_skip_counter = 0

# --- Main Loop ---
while True:
    # Wait for a frame (assuming event.framedrawn() is blocking or synchronous)
    frame_data = await event.framedrawn()
    try:
        width, height, data_bytes = frame_data
    except Exception as e:
        print("env_multi.py: Error unpacking frame data:", e)
        continue

    frame = process_frame(width, height, data_bytes)
    if frame is None:
        print("env_multi.py: Frame processing failed.")
        continue
    frame_buffer.append(frame)
    
    # Read the current action from our state dictionary.
    with state_lock:
        current_action = state["action"]
    apply_action(current_action)
    frame_skip_counter += 1

    if frame_skip_counter < frame_skip:
        continue

    # Compute and update state outcomes.
    reward, terminal, speed, lap_progress = compute_reward()
    timestep += 1
    with state_lock:
        state["timestep"] = timestep
        state["reward"] = reward
        state["terminal"] = bool(terminal)
        state["speed"] = speed
        state["lap_progress"] = lap_progress

    frame_skip_counter = 0

    if terminal:
        time.sleep(0.01)
        reset_environment(initial=False)
        continue

    # Update the frame in the state with the oldest frame.
    with state_lock:
        state["frame"] = frame_buffer[0]
