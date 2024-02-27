import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # You can change the parameter to the desired camera index

# Initialize kNN background subtractor with custom parameters
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=1200,
    detectShadows=False
)

# Set the capture duration to 6 seconds
capture_duration = 6  # in seconds

# Record the start time
start_time = time.time()

# Lists to store foreground masks over time
fg_masks = []

while time.time() - start_time < capture_duration:
    # Capture a frame
    ret, frame = cap.read()

    # Apply kNN background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Perform morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Apply median blur to the foreground mask to further reduce noise
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Store the foreground mask
    fg_masks.append(fg_mask)

    # Display the original frame and the foreground mask
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Convert the list of foreground masks to numpy array for easier calculation
fg_masks = np.array(fg_masks)

# Calculate temporal changes in the foreground masks
temporal_changes = np.sum(np.abs(np.diff(fg_masks, axis=0)), axis=(1, 2))

# Set a threshold to decide whether an object is detected based on temporal changes
detection_threshold = 500  # Adjust as needed
detected_labels = temporal_changes > detection_threshold

# Create simulated ground truth labels
ground_truth_labels = np.ones_like(detected_labels, dtype=bool)

# Calculate evaluation metrics
accuracy = accuracy_score(ground_truth_labels, detected_labels)
precision = precision_score(ground_truth_labels, detected_labels)
recall = recall_score(ground_truth_labels, detected_labels)
f1 = f1_score(ground_truth_labels, detected_labels)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
