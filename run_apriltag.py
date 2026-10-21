"""Live AprilTag 3 detection on a Raspberry Pi desktop. Press q to quit.

python run_apriltag.py
python run_apriltag.py --tag-size 0.08

Tag size is the measured outer black square side, in meters (tag36h11).
The optional pose uses the user's provisional 1920x1440 calibration.
Camera mode/crop/focus must match calibration for meaningful pose results.
"""
import argparse
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from pupil_apriltags import Detector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", default="tag36h11")
parser.add_argument("--tag-size", type=float, help="Physical tag side in meters; enables provisional pose")
args = parser.parse_args()
if args.tag_size is not None and args.tag_size <= 0:
    parser.error("Tag size must be positive.")

# Transcribed from the calibration screenshot. Not validated for accurate pose.
K = np.array([[14419.0983, 0, 975.363969], [0, 15366.7748, 719.427415], [0, 0, 1]], dtype=float)
dist = np.array([-0.320939984, 616.390811, 0.0172593286, 0.0285645086, 2.26537009])
camera_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
pose_enabled = args.tag_size is not None
detector = Detector(families=args.family, nthreads=4, quad_decimate=2.0)
camera = Picamera2()
# RGB888 in Picamera2 gives B,G,R byte order, matching OpenCV's BGR convention.
camera.configure(camera.create_still_configuration(main={"size": (1920, 1440), "format": "RGB888"}))
if pose_enabled:
    map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, K, (1920, 1440), cv2.CV_32FC1)
    print("Using provisional calibration: pose values are not validated.")
last_print = 0.0

try:
    camera.start()
    time.sleep(2)
    print(f"Detecting {args.family}. Select the preview and press q to quit.")
    while True:
        frame = camera.capture_array("main")
        if pose_enabled:
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=pose_enabled,
                               camera_params=camera_params if pose_enabled else None,
                               tag_size=args.tag_size)
        report = time.monotonic() - last_print >= 1.0
        for tag in tags:
            corners = np.rint(tag.corners).astype(np.int32)
            cv2.polylines(frame, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
            center = tuple(np.rint(tag.center).astype(int))
            cv2.putText(frame, f"ID {tag.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if pose_enabled:
                rvec, _ = cv2.Rodrigues(tag.pose_R)
                cv2.drawFrameAxes(frame, K, np.zeros(5), rvec, tag.pose_t, args.tag_size / 2)
                if report:
                    print(f"ID {tag.tag_id}, translation in camera frame [m]: {tag.pose_t.ravel()}")
                    print("Rotation matrix (tag to camera):\n", tag.pose_R)
            elif report:
                print(f"Detected ID {tag.tag_id}")
        if report:
            last_print = time.monotonic()
        cv2.putText(frame, f"Tags: {len(tags)} | q: quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("AprilTag", cv2.resize(frame, (960, 720)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()
