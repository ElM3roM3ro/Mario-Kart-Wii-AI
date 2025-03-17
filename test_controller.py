import time
import sys
sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")
# Check if we're running inside Dolphin by trying to import its controller module.
from dolphin import controller, savestate, event

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
        print("Wiimote Actual: ", controller.get_wiimote_buttons(0))
        print("Nunchuck Actual: ", controller.get_wii_nunchuk_buttons(0))
    
    print("[Test] Clearing Wii inputs.")
    controller.set_wiimote_buttons(0, {})
    controller.set_wii_nunchuk_buttons(0, {})
    print("[Test] Input test complete.")

#savestate.load_from_slot(0)
send_test_inputs()