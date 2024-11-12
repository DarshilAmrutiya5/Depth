import cv2
import numpy as np

# Create VideoCapture objects for both cameras
cap_l = cv2.VideoCapture(0)
cap_r = cv2.VideoCapture(1)

# Check if cameras opened successfully
if not cap_l.isOpened() or not cap_r.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Set up the StereoSGBM parameters
min_disp = 0
num_disp = 160  # Needs to be divisible by 16
block_size = 5
stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=block_size)

try:
    while True:
        # Capture frames
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print("Failed to capture frame.")
            break

        # Compute disparity map
        disparity = stereo.compute(frame_l, frame_r).astype(np.float32) / 16.0

        # Normalize disparity map for better visualization
        normalized_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Display images
        cv2.imshow('Left Camera', frame_l)
        cv2.imshow('Right Camera', frame_r)
        cv2.imshow('Disparity Map', normalized_disparity)

        # Relative depth estimation (simplistic approach)
        # Lower disparity values indicate closer objects
        # Higher disparity values indicate farther objects
        center_pixel_disparity = disparity[frame_l.shape[0] // 2, frame_l.shape[1] // 2]
        print("Estimated relative depth at center pixel:", center_pixel_disparity)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()