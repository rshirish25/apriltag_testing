"""Capture 320x240 checkerboard photos for camera calibration on a Pi."""

import time
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2


IMAGE_SIZE = (320, 240)
OUTPUT_DIR = Path(__file__).resolve().parent / "images_320x240"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    camera = Picamera2()
    camera.configure(camera.create_still_configuration(
        main={"size": IMAGE_SIZE, "format": "RGB888"}
    ))

    try:
        camera.start()
        time.sleep(2)
        print("Show the checkerboard at different positions and angles.")
        print("Press s to save an image, or q to quit.")

        while True:
            frame = camera.capture_array("main")
            if (frame.shape[1], frame.shape[0]) != IMAGE_SIZE:
                raise RuntimeError(f"Unexpected capture size: {frame.shape[1]}x{frame.shape[0]}")

            cv2.imshow("Calibration capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                filename = f"checkerboard_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
                path = OUTPUT_DIR / filename
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not save {path}")
                print(f"Saved {path}")
            elif key == ord("q"):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
