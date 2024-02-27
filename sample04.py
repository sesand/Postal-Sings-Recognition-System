import cv2
import os
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Set up video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up MOG2 background subtractor
history_mog2 = 100  # Adjust the history parameter as needed for MOG2
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(history=history_mog2, detectShadows=False)

# Create a folder to store the subtracted frames
output_folder = '/home/raspberry/Desktop/sesan project-1/MOG2/mog2/MOG2_WithOriginal'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the capture duration (in seconds)
capture_duration = 6
end_time = datetime.now() + timedelta(seconds=capture_duration)

frame_count = 0

# Specify the output video files
output_video_file_subtracted = '/home/raspberry/Desktop/sesan project-1/MOG2/mog2/MOG2_WithOriginal_Subtracted.mp4'
output_video_file_original = '/home/raspberry/Desktop/sesan project-1/MOG2/mog2/MOG2_WithOriginal_Original.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on your system

output_video_subtracted = cv2.VideoWriter(output_video_file_subtracted, fourcc, 20.0, (640, 480))  # Adjust frame size and FPS as needed
output_video_original = cv2.VideoWriter(output_video_file_original, fourcc, 20.0, (640, 480))  # Adjust frame size and FPS as needed

# Ground truth or reference frames for evaluation
reference_frames = []
ground_truth = []  # 0 for no hand, 1 for hand (example assumption)

while True:
    # Capture a frame
    ret, frame = video_capture.read()

    # Check if the frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Apply MOG2 background subtraction
    fgmask_mog2 = fgbg_mog2.apply(frame)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 5))
    fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Apply median filtering to further reduce noise
    fgmask_mog2 = cv2.medianBlur(fgmask_mog2, 5)

    # Find contours in the subtracted image
    contours, _ = cv2.findContours(fgmask_mog2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the original frame and the MOG2 background subtraction
    cv2.imshow('Original Frame', frame)
    cv2.imshow('MOG2 Background Subtraction', fgmask_mog2)

    # Save the subtracted frame as an image (optional)
    output_file_subtracted = os.path.join(output_folder, f'subtracted_frame_{frame_count}.png')
    cv2.imwrite(output_file_subtracted, fgmask_mog2)

    # Write frame to the subtracted video file
    output_video_subtracted.write(cv2.cvtColor(fgmask_mog2, cv2.COLOR_GRAY2BGR))  # Convert to BGR before writing

    # Write frame to the original video file
    output_video_original.write(frame)

    # Store the frame for evaluation purposes
    reference_frames.append(frame.copy())

    # Ground truth information (modify based on your data)
    # For example, assuming every other frame contains a hand
    ground_truth.append(frame_count % 2)

    frame_count += 1

    # Check if the specified duration has elapsed
    if datetime.now() >= end_time:
        break

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, close windows, and release the video writers
video_capture.release()
output_video_subtracted.release()
output_video_original.release()
cv2.destroyAllWindows()

# Evaluate MOG2 background subtraction
background_subtraction_scores = [fgbg_mog2.apply(reference_frames[i]).mean() for i in range(len(reference_frames))]

# Explore different thresholds for binary classification
thresholds = [10, 20, 30]  # Add more thresholds if needed
best_threshold = None
best_f1 = 0

for threshold in thresholds:
    # Convert scores to binary classification
    background_subtraction_predictions = [1 if score > threshold else 0 for score in background_subtraction_scores]

    # Evaluate metrics
    accuracy_bs = accuracy_score(ground_truth, background_subtraction_predictions)
    recall_bs = recall_score(ground_truth, background_subtraction_predictions)
    precision_bs = precision_score(ground_truth, background_subtraction_predictions, zero_division=0)
    f1_bs = f1_score(ground_truth, background_subtraction_predictions)

    # Print metrics for each threshold
    print(f'Threshold: {threshold}')
    print(f'Accuracy: {accuracy_bs}')
    print(f'Recall: {recall_bs}')
    print(f'Precision: {precision_bs}')
    print(f'F1 Score: {f1_bs}')

    # Update best threshold based on F1 score
    if f1_bs > best_f1:
        best_f1 = f1_bs
        best_threshold = threshold

# Print metrics for the best threshold
print(f'Background Subtraction Metrics (Best Threshold = {best_threshold}):')
print(f'Accuracy: {accuracy_bs}')
print(f'Recall: {recall_bs}')
print(f'Precision: {precision_bs}')
print(f'F1 Score: {f1_bs}')

print(f'\nCapture completed. Subtracted frames saved in: {output_folder}')
print(f'Output video file (Subtracted): {output_video_file_subtracted}')
print(f'Output video file (Original): {output_video_file_original}')
