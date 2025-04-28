"""
screen_recorder.py
Records your screen for 90 seconds, then waits until
the next hour to repeat.
"""

import time, datetime
import numpy as np
import cv2
from mss import mss

# --- CONFIG ---
FPS = 10.0
DURATION = 60          # seconds to record each time
SCALE_PERCENT = 50      # set to 100 to keep full-res, or lower to downsize
CODEC = "mp4v"          # use "XVID" for AVI, "mp4v" for MP4
# ---------------

def record_screen(duration, filename, fps, scale):
    with mss() as sct:
        mon = sct.monitors[1]
        w, h = mon["width"], mon["height"]

        # compute output size
        out_w = int(w * scale / 100)
        out_h = int(h * scale / 100)

        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        out = cv2.VideoWriter(filename, fourcc, fps, (out_w, out_h))

        start = time.time()
        while time.time() - start < duration:
            img = np.array(sct.grab(mon))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if scale != 100:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            out.write(frame)

        out.release()

def seconds_until_next_hour():
    now = datetime.datetime.now()
    next_hr = (now.replace(minute=0, second=0, microsecond=0)
               + datetime.timedelta(hours=1))
    return (next_hr - now).total_seconds()

def main():
    try:
        while True:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "mp4" if CODEC == "mp4v" else "avi"
            fname = f"screen_record_{ts}.{ext}"
            print(f"[{datetime.datetime.now()}] Recording → {fname}")
            record_screen(DURATION, fname, FPS, SCALE_PERCENT)

            wait = int(seconds_until_next_hour())
            print(f"[{datetime.datetime.now()}] Done. Sleeping {wait}s until next hour.")
            time.sleep(wait)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")

if __name__ == "__main__":
    main()
