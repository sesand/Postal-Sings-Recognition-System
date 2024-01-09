import cv2
import numpy as np
import os

# Function to perform Lucas-Kanade optical flow and save frames
def lucas_kanade_optical_flow(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    frame_files = sorted(os.listdir(input_folder))

    # Loop through each pair of consecutive frames
    for i in range(len(frame_files) - 1):
        # Read two consecutive frames
        frame1 = cv2.imread(os.path.join(input_folder, frame_files[i]), cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(os.path.join(input_folder, frame_files[i + 1]), cv2.IMREAD_GRAYSCALE)

        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of the optical flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold for motion detection (adjust as needed)
        threshold = 2.0

        # Create a binary mask based on the threshold
        motion_mask = (magnitude > threshold).astype(np.uint8) * 255

        # Save the resulting frame with detected motion
        output_frame_path = os.path.join(output_folder, f"motion_frame_{i}.png")
        cv2.imwrite(output_frame_path, motion_mask)

    print("Motion detection and saving complete.")

# Specify the input frames folder and output folder for motion frames
input_frames_folder = r"C:\Users\admin\Desktop\Postal signs\FD & BS coding\Framediffer&backgroundSubtraction\Letter"
output_motion_frames_folder = r"C:\Users\admin\Desktop\Postal signs\LucasKanade coding\LucasKanade Frames\Letter"

# Call the function to perform motion detection using Lucas-Kanade optical flow
lucas_kanade_optical_flow(input_frames_folder, output_motion_frames_folder)
