"""AprilTag detection and pose for Picamera."""
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from pupil_apriltags import Detector

# Settings you can edit.
tag_family = "tag36h11"
tag_size = 0.096       # Outer black square side length in meters.
estimate_pose = True  # Set to False for outlines and IDs only.

if estimate_pose and tag_size <= 0:
    raise ValueError("Tag size must be positive.")

# Camera calibration parameters
camera_matrix = np.array([
    [14419.0983, 0, 975.363969],
    [0, 15366.7748, 719.427415],
    [0, 0, 1],
])
distortion = np.array([-0.320939984, 616.390811, 0.0172593286, 0.0285645086, 2.26537009])
camera_parameters = (
    camera_matrix[0, 0], camera_matrix[1, 1],  # fx, fy
    camera_matrix[0, 2], camera_matrix[1, 2],  # cx, cy
)

# Prepare the AprilTag detector and Pi camera.
detector = Detector(families=tag_family, nthreads=4, quad_decimate=2.0)
camera = Picamera2()
camera.configure(camera.create_still_configuration(
    main={"size": (1920, 1440), "format": "RGB888"}
))
if estimate_pose:
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix, distortion, None, camera_matrix, (1920, 1440), cv2.CV_32FC1
    )
    print("Using provisional calibration: pose values are not validated.")
last_print_time = 0.0

try:
    camera.start()
    time.sleep(2)
    while True:
        # Capture an image, correct distortion for pose, and convert to grayscale.
        frame = camera.capture_array("main")
        if estimate_pose:
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # AprilTag performs detection and optional pose estimation here.
        tags = detector.detect(
            gray,
            estimate_tag_pose=estimate_pose,
            camera_params=camera_parameters,
            tag_size=tag_size,
        )

        # Draw each tag's outline, ID, and optional coordinate axes.
        for tag in tags:
            corners = np.rint(tag.corners).astype(np.int32)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 3)
            center = tuple(np.rint(tag.center).astype(int))
            cv2.putText(frame, f"ID {tag.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if estimate_pose:
                rotation_vector, _ = cv2.Rodrigues(tag.pose_R)
                cv2.drawFrameAxes(frame, camera_matrix, np.zeros(5), rotation_vector, tag.pose_t, tag_size / 2)

        # Print results once per second, while detection continues every frame.
        if time.monotonic() - last_print_time >= 1.0:
            for tag in tags:
                if estimate_pose:
                    print(f"ID {tag.tag_id}, translation in camera frame [m]: {tag.pose_t.ravel()}")
                    print("Rotation matrix (tag to camera):\n", tag.pose_R)
                else:
                    print(f"Detected ID {tag.tag_id}")
            last_print_time = time.monotonic()

        # Display the preview. Press q while this window is selected to stop.
        cv2.imshow("AprilTag", cv2.resize(frame, (960, 720)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()
