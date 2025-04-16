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
        print("state info error")
        return 0.0, False, 0.0, 0.0
    speed = state_info['speed']
    lap_progress = state_info['lap_progress']
    rewardN = 0.0
    terminalN = False
    if last_lap_progress is None:
        last_lap_progress = lap_progress
    lap_diff = lap_progress - last_lap_progress
    if lap_diff >= 0.01:
        num_increments = int(lap_diff / 0.01)
        rewardN += 1.0 * num_increments
        if int(lap_progress) > int(last_lap_progress) and int(last_lap_progress) != 0:
            rewardN += 3.3
        last_lap_progress = lap_progress
    if speed < 65:
        rewardN -= 10.0
        terminalN = True
    elif speed > 90:
        rewardN += 0.025
    if lap_progress >= 4.0:
        rewardN += 10.0
        terminalN = True
    return rewardN, terminalN, speed, lap_progress

# --- New Global Variables for Drift Action Tracking ---
prev_action = None         # Tracks the previous action selected.
drift_counter = 0          # Counts the number of frames a drift action was held.
drift_actions = {1, 2, 3, 4} # Actions considered to be drift actions.

def apply_action(action_index):
    """
    Map the provided action index to controller commands with the new action modifications:
      0: Accelerate straight (as before).
      1: Drift left 45° (diagonal up-left).
      2: Drift right 45° (diagonal up-right).
      3: Hard drift left (unchanged hard input).
      4: Hard drift right (unchanged hard input).
      5: No-op (only holding A button).
      7: Special action (remains unchanged).
    Any undefined action defaults to a no-op.
    """
    if action_index == 0:
        # Accelerate straight:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        controller.set_wiimote_swing(0, 0.0, 1.0, 0.0, 0.5, 16.0, 2.0, 0.0)
        nunchuck_action = {"StickX": 0.0, "StickY": 1.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 1:
        # Drift left 45° (diagonal up-left):
        controller.set_wiimote_buttons(0, {"A": True, "B": True})
        nunchuck_action = {"StickX": -1.0, "StickY": 1.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 2:
        # Drift right 45° (diagonal up-right):
        controller.set_wiimote_buttons(0, {"A": True, "B": True})
        nunchuck_action = {"StickX": 1.0, "StickY": 1.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 3:
        # Hard drift left:
        controller.set_wiimote_buttons(0, {"A": True, "B": True})
        nunchuck_action = {"StickX": -1.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 4:
        # Hard drift right:
        controller.set_wiimote_buttons(0, {"A": True, "B": True})
        nunchuck_action = {"StickX": 1.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 5:
        # No-op: only holding A button (neutral stick):
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    elif action_index == 6:
        # Special action remains unchanged:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": True}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        #nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        #controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return
    else:
        # Fallback action for any undefined index:
        controller.set_wiimote_buttons(0, {"A": True, "B": False})
        nunchuck_action = {"StickX": 0.0, "StickY": 0.0, "Z": False}
        controller.set_wii_nunchuk_buttons(0, nunchuck_action)
        return

def reset_environment(initial=False):
    """
    Reset game state and clear the frame stack.
    In addition, set the new_episode flag so that the next call to process_frame will reinitialize the state.
    """
    global last_lap_progress, frame_stack, last_reward, frame_num, new_episode, drift_counter
    last_lap_progress = None
    last_reward = 0
    frame_num = 0
    frame_stack.clear()
    new_episode = True  # Force reinitialization of the frame stack on the next frame.
    drift_counter = 0
    if not initial:
        reset_choice = random.randint(1, 4)
        reset_choice = 1
        if reset_choice == 1:
            savestate.load_from_file(r"E:\MKWii_Savestates\funky_flame_delfino_savestate_startv2.sav")
        elif reset_choice == 2:
            savestate.load_from_file(r"E:\MKWii_Savestates\funky_flame_delfino_savestate2.sav")
        elif reset_choice == 3:
            savestate.load_from_file(r"E:\MKWii_Savestates\funky_flame_delfino_savestate3.sav")
        elif reset_choice == 4:
            savestate.load_from_file(r"E:\MKWii_Savestates\funky_flame_delfino_savestate4.sav")

##############################################################################
# Main loop with persistent connection, asynchronous awaits, frameskip, sequential stacking,
# and drift penalty logic.
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

while True:
    try:
        command = conn.recv()
    except Exception as e:
        print("env_multi.py: Error receiving command:", e)
        break

    if isinstance(command, dict):
        cmd = command.get("command")
        if cmd == "step":
            action = command.get("action", 0)

            # --- Drift Penalty Check ---
            # If the previous action was a drift and the new action is either not drifting
            # or is a different drift than before, and the drift duration was less than 14 frames,
            # apply a penalty of 0.1 to the reward.
            penalty = 0.0
            if prev_action is not None and (prev_action in drift_actions):
                if (action not in drift_actions) or (action in drift_actions and action != prev_action):
                    if drift_counter < 14:
                        penalty = 0.1

            reward = 0.0
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
                terminal = terminal or terminalN
                reward += rewardN

                if terminal:
                    # Terminal branch: advance extra frames and reset environment.
                    for j in range(2):
                        #apply_action(action)
                        await event.frameadvance()
                    reset_environment(initial=False)
                    for j in range(1):
                        #apply_action(action)
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

            # Apply the drift penalty if applicable.
            reward -= penalty

            # Update drift_counter based on the current action.
            if action in drift_actions:
                if prev_action == action:
                    drift_counter += frame_skip
                else:
                    drift_counter = frame_skip
            else:
                drift_counter = 0
            prev_action = action

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
        elif cmd == "reset":
            # Handle a reset command here:
            reset_environment(initial=False)
            
            # Get a new frame after reset:
            frame_data = await event.framedrawn()
            try:
                width, height, data_bytes = frame_data
            except Exception as e:
                print("env_multi.py: Error unpacking frame data after reset:", e)
                continue
            raw_img = Image.frombytes('RGB', (width, height), data_bytes, 'raw')
            # Use terminal=True or False depending on how you wish to signal the reset.
            new_obs = process_frame(raw_img, terminal=True)
            
            # Send back the reset response with new initial observation.
            conn.send({"command": "reset", "observation": np.copy(new_obs)})
        else:
            print("env_multi.py: Received unknown command.")
    else:
        print("env_multi.py: Received unknown command.")
