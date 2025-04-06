import time
import collections
import sys
# Set up site–packages path.
user = "Zach"
if user == "Nolan":
    sys.path.append(r"C:\Users\nolan\AppData\Local\Programs\Python\Python312\Lib\site-packages")
elif user == "Zach":
    sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")
elif user == "Victor":
    sys.path.append(r"C:\Users\victo\AppData\Local\Programs\Python\Python312\Lib\site-packages")
import os
import random
import numpy as np
from PIL import Image
import torch
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

# --- Global Variables and Helpers ---
state_lock = Lock()
frame_buffer = collections.deque(maxlen=4)
frame_skip = 4
target_width = 128
target_height = 128
last_lap_progress = None

def process_frame(width, height, data_bytes):
    try:
        image = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    except Exception as e:
        print("env_multi.py: Error in Image.frombytes:", e)
        return None
    image = image.convert("L").resize((target_width, target_height), Image.BILINEAR)
    return np.array(image)

def read_game_state():
    try:
        speed = memory.read_f32(0x80fad2c4)
        lap_progress = memory.read_f32(0x80e48d3c)
        current_lap = int(memory.read_f32(0x80e96428))
        max_lap = int(memory.read_f32(0x80e96428))
        return {
            'speed': speed,
            'lap_progress': lap_progress,
            'current_lap': current_lap,
            'max_lap': max_lap
        }
    except Exception as e:
        print("env_multi.py: Error reading game state:", e)
        return None

def compute_reward():
    global last_lap_progress
    state_info = read_game_state()
    if state_info is None:
        return 0.0, False, 0.0, 0.0
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
    if lap_progress >= 4.0:
        reward += 10.0
        terminal = True
    return reward, terminal, speed, lap_progress

def apply_action(action_index):
    # Maps the provided action index to controller commands.
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
    global last_lap_progress, frame_buffer
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

##############################################################################
# Main loop with persistent connection, synchronous awaits, and frame-skip action holding.
##############################################################################
port = int(os.environ.get("ENV_PORT", "6000"))
address = ('localhost', port)
authkey = b'secret'
listener = Listener(address, authkey=authkey)
print(f"env_multi.py: Listening on {address} for persistent connection.")
conn = listener.accept()
print("env_multi.py: Persistent connection established.")

# Reset environment and build initial observation.
reset_environment(initial=True)
# Collect 4 frames to form the initial observation.
while len(frame_buffer) < 4:
    frame_data = await event.framedrawn()
    try:
        width, height, data_bytes = frame_data
    except Exception as e:
        print("env_multi.py: Error unpacking frame data:", e)
        continue
    frame = process_frame(width, height, data_bytes)
    if frame is not None:
        frame_buffer.append(frame)
current_obs = np.stack(list(frame_buffer), axis=0)
# Send the initial (reset) observation to the trainer.
conn.send({"command": "reset", "observation": current_obs})

# Main loop: wait for a "step" command and then process it.
while True:
    try:
        command = conn.recv()
    except Exception as e:
        print("env_multi.py: Error receiving command:", e)
        break

    if isinstance(command, dict) and command.get("command") == "step":
        action = command.get("action", 0)
        # Apply the action once before starting the frame skip loop.
        apply_action(action)

        # Collect a new set of frames while ensuring that the same action persists
        # throughout the skipped frames.
        frame_buffer.clear()
        frame_skip_counter = 0
        while frame_skip_counter < frame_skip:
            # Reapply the same action to hold it constant.
            apply_action(action)
            frame_data = await event.framedrawn()
            try:
                width, height, data_bytes = frame_data
            except Exception as e:
                print("env_multi.py: Error unpacking frame data:", e)
                continue
            frame = process_frame(width, height, data_bytes)
            if frame is None:
                continue
            frame_buffer.append(frame)
            frame_skip_counter += 1

        # Compute reward and terminal flag.
        reward, terminal, speed, lap_progress = compute_reward()
        next_obs = np.stack(list(frame_buffer), axis=0)
        transition = {
            "observation": current_obs,
            "action": action,
            "reward": reward,
            "next_observation": next_obs,
            "terminal": terminal,
            "speed": speed,
            "lap_progress": lap_progress
        }

        # Terminal handling (RL best practice):
        # The terminal flag is part of the transition so that the agent knows
        # the episode ended on this transition. Immediately after, the environment
        # is reset to provide a new starting observation for the next episode.
        if terminal:
            reset_environment(initial=False)
            frame_buffer.clear()
            frame_skip_counter = 0
            # Collect 4 frames for the new initial observation.
            while frame_skip_counter < 5:
                frame_data = await event.framedrawn()
                try:
                    width, height, data_bytes = frame_data
                except Exception as e:
                    print("env_multi.py: Error unpacking frame data during reset:", e)
                    continue
                frame = process_frame(width, height, data_bytes)
                if frame is None:
                    continue
                frame_buffer.append(frame)
                frame_skip_counter += 1
            next_obs = np.stack(list(frame_buffer), axis=0)
            transition["next_observation"] = next_obs

        # Update current observation and send the full transition back.
        current_obs = next_obs
        conn.send(transition)
    else:
        conn.send({"error": "Unknown command"})