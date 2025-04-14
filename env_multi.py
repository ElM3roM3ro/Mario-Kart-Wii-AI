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

# Parameters for frameskipping and stacking.
frame_skip = 4           # Number of frames to process in the frameskip loop.
frame_stack_size = 4     # Number of processed frames to stack as state.
target_width = 128       # Consider reducing to 84 if you encounter performance issues.
target_height = 128

# Global frame stack used to form the state.
frame_stack = collections.deque(maxlen=frame_stack_size)
last_lap_progress = None   # For reward computation.
last_reward = 0            # Accumulated reward over frameskip.
frame_num = 0              # Frame counter.

# New flag indicating that the next frame should be treated as a new episode.
new_episode = True

def reset_frame_stack(processed_frame):
    """
    Clears the current frame stack and fills it with copies of the provided processed_frame.
    Returns the newly constructed frame stack (as a numpy array).
    """
    frame_stack.clear()
    for _ in range(frame_stack_size):
        frame_stack.append(processed_frame.copy())
    return np.stack(frame_stack, axis=0)

def process_frame(raw_img, terminal=False):
    """
    Process the raw PIL image by converting it to grayscale and resizing it.
    If terminal is True OR if this is the first frame of a new episode (indicated by new_episode flag)
    then the frame stack is reinitialized by duplicating the processed frame.
    Otherwise, the new frame is appended to the existing stack.
    """
    global new_episode
    try:
        image = raw_img.convert("L").resize((target_width, target_height), Image.BILINEAR)
    except Exception as e:
        print("env_multi.py: Error processing frame:", e)
        return None

    processed_frame = np.array(image)

    if terminal:
        # In terminal cases, reinitialize the stack with the terminal frame.
        return reset_frame_stack(processed_frame)
    else:
        # If this is the first frame of a new episode or the stack is empty, reinitialize it.
        if new_episode or len(frame_stack) == 0:
            new_state = reset_frame_stack(processed_frame)
            new_episode = False  # Clear the flag after reinitialization.
            return new_state
        else:
            frame_stack.append(processed_frame.copy())
            return np.stack(frame_stack, axis=0)

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
    """
    Compute and return (rewardN, terminalN, speed, lap_progress) for the current frame.
    """
    global last_lap_progress
    state_info = read_game_state()
    if state_info is None:
        return 0.0, False, 0.0, 0.0
    speed = state_info['speed']
    lap_progress = state_info['lap_progress']
    rewardN = 0.0
    terminalN = False
    if last_lap_progress is None:
        last_lap_progress = lap_progress
    lap_diff = lap_progress - last_lap_progress
    if lap_diff >= 0.001:
        num_increments = int(lap_diff / 0.001)
        rewardN += 0.1 * num_increments
        if int(lap_progress) > int(last_lap_progress) and int(last_lap_progress) != 0:
            rewardN += 3.0
        last_lap_progress = lap_progress
    if speed < 45:
        rewardN -= 10.0
        terminalN = True
    if lap_progress >= 4.0:
        rewardN += 10.0
        terminalN = True
    return rewardN, terminalN, speed, lap_progress

def apply_action(action_index):
    # Map the provided action index to controller commands.
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
    """
    Reset game state and clear the frame stack.
    In addition, set the new_episode flag so that the next call to process_frame will reinitialize the state.
    """
    global last_lap_progress, frame_stack, last_reward, frame_num, new_episode
    last_lap_progress = None
    last_reward = 0
    frame_num = 0
    frame_stack.clear()
    new_episode = True  # Force reinitialization of the frame stack on the next frame.
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
# Main loop with persistent connection, asynchronous awaits, frameskip, and sequential stacking.
##############################################################################
port = int(os.environ.get("ENV_PORT", "6000"))
address = ('localhost', port)
authkey = b'secret'
listener = Listener(address, authkey=authkey)
print(f"env_multi.py: Listening on {address} for persistent connection.")
conn = listener.accept()
print("env_multi.py: Persistent connection established.")

# --- Build the Initial Observation ---
reset_environment(initial=True)
initial_obs = None
while initial_obs is None:
    frame_data = await event.framedrawn()
    try:
        width, height, data_bytes = frame_data
    except Exception as e:
        print("env_multi.py: Error unpacking frame data:", e)
        continue
    raw_img = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
    initial_obs = process_frame(raw_img, terminal=False)
conn.send({"command": "reset", "observation": np.copy(initial_obs)})

# --- Main Loop ---
while True:
    try:
        command = conn.recv()
    except Exception as e:
        print("env_multi.py: Error receiving command:", e)
        break

    if isinstance(command, dict) and command.get("command") == "step":
        action = command.get("action", 0)
        #apply_action(action)
        reward = 0
        terminal = False

        # Frameskip loop: process multiple frames and accumulate rewards.
        for i in range(frame_skip):
            apply_action(action)
            if i == frame_skip - 1:
                # On the last frame, draw a new frame.
                frame_data = await event.framedrawn()
                try:
                    width, height, data_bytes = frame_data
                except Exception as e:
                    print("env_multi.py: Error unpacking frame data:", e)
                    continue
            else:
                await event.frameadvance()

            # Compute per-frame reward and terminal flag.
            rewardN, terminalN, speed, lap_progress = compute_reward()
            if not terminal:
                terminal = terminal or terminalN
                reward += rewardN

            if terminal:
                # Terminal branch: advance extra frames and reset environment.
                for j in range(2):
                    apply_action(action)
                    await event.frameadvance()
                reset_environment(initial=False)
                for j in range(1):
                    apply_action(action)
                    await event.frameadvance()
                frame_data = await event.framedrawn()
                try:
                    width, height, data_bytes = frame_data
                except Exception as e:
                    print("env_multi.py: Error unpacking frame data after terminal:", e)
                    continue
                # Process the frame with terminal=True to create a terminal observation.
                new_obs = process_frame(Image.frombytes('RGB', (width, height), data_bytes, 'raw'), terminal=True)
                break  # Exit frameskip loop on terminal.
            frame_num += 1

        # If not terminal, process the final frame normally.
        if not terminal:
            raw_img = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
            new_obs = process_frame(raw_img, terminal=False)

        transition = {
            "observation": np.copy(initial_obs),
            "action": action,
            "reward": reward,
            "next_observation": np.copy(new_obs),
            "terminal": terminal,
            "speed": speed,
            "lap_progress": lap_progress
        }

        conn.send(transition)
        # For the next step, update the current observation.
        initial_obs = new_obs
