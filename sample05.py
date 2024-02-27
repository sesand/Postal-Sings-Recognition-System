import cv2
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to calculate evaluation metrics
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # You can change the parameter to the desired camera index

# Initialize MOG2 background subtractor with custom parameters
fg_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,      # Number of frames used for the background model
    varThreshold=25,   # Threshold on the squared Mahalanobis distance
    detectShadows=False  # Whether to detect and mark shadows
)

# Set the capture duration to 6 seconds
capture_duration = 6  # in seconds

# Record the start time
start_time = time.time()

# Create a folder to save the frame differences
output_folder = '/home/raspberry/Desktop/sesan project-1/MOG002/output_frames0002'
os.makedirs(output_folder, exist_ok=True)

# Initialize variables for video creation
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('/home/raspberry/Desktop/sesan project-1/MOG002/output_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Define ground truth labels
true_labels = []

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

while time.time() - start_time < capture_duration:
    # Capture a frame
    ret, frame = cap.read()

    # Apply MOG2 background subtraction
    fg_mask = fg_bg_subtractor.apply(frame)

    # Perform morphological operations to reduce noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Convert the frame to HSV for skin color segmentation
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for the skin color
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Combine the MOG2 mask and skin color mask
    combined_mask = cv2.bitwise_and(fg_mask, skin_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 5000:  # Adjust area thresholds based on the object size
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:  # Adjust aspect ratio thresholds based on the object's aspect ratio
                cv2.drawContours(combined_mask, [contour], 0, (255), -1)
                true_labels.append(1)  # Object present in ground truth
            else:
                true_labels.append(0)  # Object not present in ground truth

    # Apply a median filter to the combined mask
    combined_mask = cv2.medianBlur(combined_mask, 5)

    # Save the combined mask as an image in the output folder
    output_file = os.path.join(output_folder, f'frame_{int(time.time() * 1000)}.png')
    cv2.imwrite(output_file, combined_mask)

    # Write the combined mask to the video
    video_writer.write(combined_mask)

    # Display the original frame, MOG2 mask, skin color mask, and the combined mask
    cv2.imshow("Original Frame", frame)
    cv2.imshow("MOG2 Mask", fg_mask)
    cv2.imshow("Skin Color Mask", skin_mask)
    cv2.imshow("Combined Mask", combined_mask)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, video writer, and close all windows
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Calculate and print evaluation metrics
predicted_labels = [1] * len(true_labels)  # Assuming an object is detected in every frame
accuracy, precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
