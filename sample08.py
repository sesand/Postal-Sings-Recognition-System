import cv2
import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # You can change the parameter to the desired camera index

# Initialize variables for frame differencing
prev_frame = None

# Initialize the MOG2 background subtractor with custom parameters
mog2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Set the capture duration to 6 seconds
capture_duration = 6  # in seconds

# Record the start time
start_time = time.time()

# Create a folder to store separated frames
output_frame_folder = '/home/raspberry/Desktop/sesan project-1/MOG002/output_frames/letter_frames004'
os.makedirs(output_frame_folder, exist_ok=True)

# Create video writer for final combined video
combined_video_writer = cv2.VideoWriter('/home/raspberry/Desktop/sesan project-1/MOG002/output_frames/letter_video004.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

# Initialize variables for evaluation metrics
true_labels = []
predicted_labels = []

frame_counter = 0

while time.time() - start_time < capture_duration:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If it's the first frame, store it for future differencing
    if prev_frame is None:
        prev_frame = gray_frame
        continue

    # Compute the absolute difference between the current and previous frame
    frame_diff = cv2.absdiff(prev_frame, gray_frame)

    # Threshold the difference to get binary image
    _, threshold_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Apply MOG2 background subtraction
    mog2_mask = mog2_bg_subtractor.apply(frame)

    # Find contours in the binary image
    contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (you can adjust the threshold)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Create a mask for the filtered contours
    contour_mask = np.zeros_like(threshold_diff)
    cv2.drawContours(contour_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the contour mask to the MOG2 result
    mog2_mask_filtered = cv2.bitwise_and(mog2_mask, contour_mask)

    # Morphological Opening to remove small noise
    kernel_opening = np.ones((5, 5), np.uint8)
    mog2_mask_filtered = cv2.morphologyEx(mog2_mask_filtered, cv2.MORPH_OPEN, kernel_opening)

    # Morphological Closing to close gaps in between fingers
    kernel_closing = np.ones((15, 15), np.uint8)
    mog2_mask_filtered = cv2.morphologyEx(mog2_mask_filtered, cv2.MORPH_CLOSE, kernel_closing)

    # Save consecutive frames
    output_frame_path = os.path.join(output_frame_folder, f'output_frame_{frame_counter}.png')
    cv2.imwrite(output_frame_path, mog2_mask_filtered)

    # Write frames to the final combined video
    combined_video_writer.write(mog2_mask_filtered)

    # Update the previous frame for the next iteration
    prev_frame = gray_frame

    # Ground truth (assuming you have a ground truth)
    # Replace this with your own logic to get the ground truth for each frame
    ground_truth = 1  # For example, if you have a ground truth that an object is present

    # Evaluate metrics
    true_labels.append(ground_truth)
    predicted_labels.append(np.sum(mog2_mask_filtered > 0))

    # Display the combined video in real-time
    cv2.imshow('Combined Video', mog2_mask_filtered)

    # Increment frame counter
    frame_counter += 1

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate evaluation metrics using scikit-learn
precision = precision_score(true_labels, (np.array(predicted_labels) > 0).astype(int), average='binary')
recall = recall_score(true_labels, (np.array(predicted_labels) > 0).astype(int), average='binary')
accuracy = accuracy_score(true_labels, (np.array(predicted_labels) > 0).astype(int))
f1_score_sklearn = f1_score(true_labels, (np.array(predicted_labels) > 0).astype(int), average='binary')

print("Metrics calculated using scikit-learn:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1_score_sklearn:.2f}")

# Release the video capture object and video writer
cap.release()
combined_video_writer.release()
cv2.destroyAllWindows()
