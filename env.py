import sys
import time
import collections
import numpy as np
from PIL import Image

# If needed, append your site-packages path so Pillow etc. is found:
# sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")

from dolphin import event, controller

class MarioKartDolphinEnv:
    """
    Environment that:
     - Receives frames from Dolphin via callback
     - Converts each frame to 128x128 grayscale
     - Stacks 4 frames
     - Defines a discrete action set for Time Trials
    """
    def __init__(self, target_width=128, target_height=128):
        self.target_width = target_width
        self.target_height = target_height

        # We'll keep the last 4 grayscale frames, each shape (128,128)
        self.frame_buffer = collections.deque(maxlen=4)

        # Subscribe to Dolphin's frame-drawn event
        event.on_framedrawn(self.on_framedrawn)

        # Discrete action set for Time Trials (example GC mapping).
        # You can tweak stick angles for 'soft', 'medium', 'hard' drift, etc.
        self.actions = [
            {"A": True, "StickX": 0.0},          # 0: Accelerate straight
            {"A": True, "StickX": -0.6},         # 1: Turn Left + Accelerate
            {"A": True, "StickX": 0.6},          # 2: Turn Right + Accelerate
            {"A": True, "R": True, "StickX": -0.3},  # 3: Drift + Soft Left
            {"A": True, "R": True, "StickX": 0.3},   # 4: Drift + Soft Right
            {"A": True, "R": True, "StickX": -0.6},  # 5: Drift + Medium Left
            {"A": True, "R": True, "StickX": 0.6},   # 6: Drift + Medium Right
            {"A": True, "R": True, "StickX": -1.0},  # 7: Drift + Hard Left
            {"A": True, "R": True, "StickX": 1.0},   # 8: Drift + Hard Right
            {"A": True, "Up": True},                 # 9: Wheelie (may depend on config)
            {"A": True, "X": True},                  # 10: Use Mushroom (Item)
        ]
        self.action_space_size = len(self.actions)

        # Internal counters
        self.current_step = 0
        self.max_steps = 2000  # end an episode after these many steps (placeholder)

    def on_framedrawn(self, width: int, height: int, data: bytes):
        """
        Callback from Dolphin each time it draws a frame.
        Convert raw bytes -> 128x128 grayscale np array -> store in frame_buffer.
        """
        # Convert raw data to a PIL image
        image = Image.frombytes('RGB', (width, height), data, 'raw')
        # Grayscale and resize
        image = image.convert("L").resize(
            (self.target_width, self.target_height),
            Image.BILINEAR
        )
        # shape: (128,128) as a NumPy array
        frame_array = np.array(image)
        self.frame_buffer.append(frame_array)

    def reset(self):
        """
        Reset environment state, clear buffer, wait for at least 4 frames, return stacked frames.
        Possibly also do a savestate load or reposition the kart, etc.
        """
        self.current_step = 0
        self.frame_buffer.clear()

        # Ensure we release all buttons at reset
        controller.set_gc_buttons(0, {})

        # Wait for frames to accumulate
        time.sleep(1.0)
        while len(self.frame_buffer) < 4:
            time.sleep(0.1)

        return self._get_stacked_frames()

    def step(self, action_idx):
        """
        Execute the chosen action on the GC controller for one frame,
        then return (obs, reward, done, info).
        """
        # 1) Apply the chosen action, if valid
        if 0 <= action_idx < len(self.actions):
            gc_input = self.actions[action_idx]
        else:
            gc_input = {}
        controller.set_gc_buttons(0, gc_input)

        # 2) Wait ~ one frame
        time.sleep(0.03)

        # 3) Construct the observation
        obs = self._get_stacked_frames()

        # 4) Placeholder reward logic
        reward = 0.0

        # 5) Check if done
        self.current_step += 1
        done = (self.current_step >= self.max_steps)

        info = {}
        return obs, reward, done, info

    def _get_stacked_frames(self):
        """
        Return a NumPy array of shape (4, 128, 128) in grayscale.
        If fewer than 4 frames available, replicate earliest.
        """
        if len(self.frame_buffer) < 4:
            frames = list(self.frame_buffer)
            while len(frames) < 4:
                frames.insert(0, frames[0])
        else:
            frames = list(self.frame_buffer)

        stacked = np.stack(frames, axis=0)  # shape (4, 128, 128)

        # If you want float in [0,1], do: stacked = stacked.astype(np.float32) / 255.0
        return stacked
