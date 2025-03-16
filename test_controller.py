# import time
# import subprocess
# import os
# import sys

# # Try to import the Dolphin controller module.
# try:
#     from dolphin import controller
# except ImportError:
#     print("Dolphin modules not found. Using dummy objects for development.")
#     class Dummy:
#         def __getattr__(self, name):
#             return lambda *args, **kwargs: None
#     controller = Dummy()

# def send_test_inputs(duration=20):
#     """
#     Cycle through several test actions for a given duration.
#     Each action is held for 2 seconds.
#     """
#     test_actions = [
#         {"A": True, "StickX": 0.0},           # Accelerate straight
#         {"A": True, "StickX": -0.5},           # Turn left
#         {"A": True, "StickX": 0.5},            # Turn right
#         {"A": True, "R": True, "StickX": -0.3},  # Drift left
#         {"A": True, "R": True, "StickX": 0.3},   # Drift right
#         {"A": True, "Up": True},                # Wheelie
#         {"A": True, "X": True},                 # Use Mushroom
#     ]
    
#     start_time = time.time()
#     action_index = 0
#     print("[Test] Starting controller input test for", duration, "seconds.")
#     while time.time() - start_time < duration:
#         action = test_actions[action_index % len(test_actions)]
#         print(f"[Test] Sending action: {action}")
#         controller.set_gc_buttons(0, action)
#         time.sleep(2)  # Hold each action for 2 seconds.
#         action_index += 1
#     print("[Test] Clearing controller inputs.")
#     controller.set_gc_buttons(0, {})
#     print("[Test] Controller input test completed.")

# def launch_dolphin():
#     """
#     Launch Dolphin with this test_controller script as the attached script.
#     Sets an environment variable so that the launched Dolphin instance does not
#     re-run the launcher portion.
#     """
#     # Update these paths to match your system.
#     dolphin_path = r"C:\Users\nolan\source\repos\dolphin\Binary\x64\Dolphin.exe"  # Path to Dolphin.exe
#     # Use this file's absolute path as the script path.
#     script_path = os.path.abspath(__file__)
#     savestate_path = r"C:\Users\nolan\OneDrive\Desktop\School\CS\Capstone\Mario-Kart-Wii-AI\delfino_savestate.sav"  # Savestate file at countdown
#     game_path = r"C:\Users\nolan\source\repos\dolphin\Source\Core\DolphinQt\MarioKart(Compress).iso"                 # Your game file

#     cmd = (
#         f'"{dolphin_path}" --no-python-subinterpreters '
#         f'--script "{script_path}" '
#         f'--save_state="{savestate_path}" '
#         f'--exec="{game_path}"'
#     )
    
#     # Copy current environment and set RUNNING_IN_DOLPHIN=1
#     env = os.environ.copy()
#     env["RUNNING_IN_DOLPHIN"] = "1"
    
#     print("[Launcher] Launching Dolphin with command:")
#     print(cmd)
#     subprocess.Popen(cmd, shell=True)
#     print("[Launcher] Dolphin launched. Waiting a few seconds for it to load...")
#     time.sleep(10)

# if __name__ == "__main__":
#     # If not running inside Dolphin, launch Dolphin and exit.
#     if "RUNNING_IN_DOLPHIN" not in os.environ:
#         print("[Launcher] Not running inside Dolphin. Launching Dolphin now...")
#         launch_dolphin()
#         print("[Launcher] Exiting launcher. Please switch to the Dolphin window.")
#         sys.exit(0)
#     else:
#         # Running inside Dolphin.
#         print("[Test] Running inside Dolphin.")
#         # Wait for a delay to ensure the savestate has fully loaded.
#         load_delay = 20  # seconds; adjust as needed
#         print(f"[Test] Waiting {load_delay} seconds for the game to load...")
#         time.sleep(load_delay)
#         print("[Test] Starting controller input test.")
#         send_test_inputs(duration=20)
#         print("[Test] Test complete. Quitting Dolphin.")
#         os._exit(0)

import time
import subprocess
import os
import sys

# Check if we're running inside Dolphin by trying to import its controller module.
from dolphin import controller

def send_test_inputs():
    """
    Cycle through a set of test input sequences using Wii remote and Wii nunchuk.
    Each sequence is held for 2 seconds.
    """
    # Define a list of input sequences.
    # For Wii remote, we use keys like "A" and "B".
    # For Wii nunchuk, we use "StickX" and "StickY" (values range from -1 to 1).
    sequences = [
        # Accelerate straight.
        {
            "wiimote": {"A": True},
            "nunchuk": {"StickX": 0.0, "StickY": 1.0}
        },
        # Steer left.
        {
            "wiimote": {"A": True},
            "nunchuk": {"StickX": -0.5, "StickY": 1.0}
        },
        # Steer right.
        {
            "wiimote": {"A": True},
            "nunchuk": {"StickX": 0.5, "StickY": 1.0}
        },
        # Drift left.
        {
            "wiimote": {"A": True, "B": True},
            "nunchuk": {"StickX": -0.5, "StickY": 1.0}
        },
        # Drift right.
        {
            "wiimote": {"A": True, "B": True},
            "nunchuk": {"StickX": 0.5, "StickY": 1.0}
        },
    ]
    
    print("[Test] Starting Wii input test.")
    for seq in sequences:
        print("[Test] Sending sequence:", seq)
        controller.set_wiimote_buttons(0, seq["wiimote"])
        controller.set_wii_nunchuk_buttons(0, seq["nunchuk"])
        time.sleep(3)
    
    print("[Test] Clearing Wii inputs.")
    controller.set_wiimote_buttons(0, {})
    controller.set_wii_nunchuk_buttons(0, {})
    print("[Test] Input test complete. Exiting Dolphin.")

if __name__ == "__main__":
    print("[Test] Running inside Dolphin. Waiting 5 seconds for the savestate to load...")
    time.sleep(5)
    send_test_inputs()