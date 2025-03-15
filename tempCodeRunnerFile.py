import time
import subprocess
import os
import sys

# Try to import the Dolphin controller module.
try:
    from dolphin import controller
except ImportError:
    print("Dolphin modules not found. Make sure this script is run by the Felk Dolphin.")
    sys.exit(1)

def send_test_inputs(duration=20):
    """
    Cycle through several test actions for a given duration.
    Each action is held for 2 seconds.
    """
    test_actions = [
        {"A": True, "StickX": 0.0},           # Accelerate straight
        {"A": True, "StickX": -0.5},           # Turn left
        {"A": True, "StickX": 0.5},            # Turn right
        {"A": True, "R": True, "StickX": -0.3},  # Drift left
        {"A": True, "R": True, "StickX": 0.3},   # Drift right
        {"A": True, "Up": True},                # Wheelie
        {"A": True, "X": True},                 # Use Mushroom
    ]
    
    start_time = time.time()
    action_index = 0
    print("[Test] Starting controller input test for", duration, "seconds.")
    while time.time() - start_time < duration:
        action = test_actions[action_index % len(test_actions)]
        print(f"[Test] Sending action: {action}")
        controller.set_gc_buttons(0, action)
        time.sleep(2)  # Hold each action for 2 seconds.
        action_index += 1
    print("[Test] Clearing controller inputs.")
    controller.set_gc_buttons(0, {})
    print("[Test] Controller input test completed.")

def launch_dolphin():
    """
    Launch Dolphin with this test_controller script as the attached script.
    Sets an environment variable so that the launched Dolphin instance does not
    re-run the launcher portion.
    """
    # Update these paths to match your system.
    dolphin_path = r"F:\DolphinEmuFork_src\dolphin\Binary\x64\Dolphin.exe"  # Path to Dolphin.exe
    # Use this file's absolute path as the script path.
    script_path = os.path.abspath(__file__)
    savestate_path = r"F:\MKWii_Capstone_Project\Mario-Kart-Wii-AI\delfino_savestate.sav"  # Savestate file at countdown
    game_path = r"E:\Games\Dolphin Games\MarioKart(Compress).iso"                 # Your game file

    cmd = (
        f'"{dolphin_path}" --no-python-subinterpreters '
        f'--script "{script_path}" '
        f'--save_state="{savestate_path}" '
        f'--exec="{game_path}"'
    )
    
    # Copy current environment and set RUNNING_IN_DOLPHIN=1
    env = os.environ.copy()
    env["RUNNING_IN_DOLPHIN"] = "1"
    
    print("[Launcher] Launching Dolphin with command:")
    print(cmd)
    subprocess.Popen(cmd, shell=True, env=env)
    print("[Launcher] Dolphin launched. Waiting a few seconds for it to load...")
    time.sleep(10)

if __name__ == "__main__":
    # When running outside Dolphin, RUNNING_IN_DOLPHIN is not set.
    if os.environ.get("RUNNING_IN_DOLPHIN") is None:
        print("[Launcher] Not running inside Dolphin. Launching Dolphin now...")
        launch_dolphin()
        print("[Launcher] Exiting launcher. Please switch to the Dolphin window.")
    else:
        print("[Test] Running inside Dolphin. Starting test_controller inputs.")
        send_test_inputs(duration=20)
