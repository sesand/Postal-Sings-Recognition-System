import cv2
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

def process_video(input_video_path, output_frame_folder, output_video_path):
    # Initialize variables for frame differencing
    prev_frame = None

    # Initialize the MOG2 background subtractor with custom parameters
    mog2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # Create video capture object for the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video details
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create video writer for final combined video
    combined_video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Initialize variables for evaluation metrics
    true_labels = []
    predicted_labels = []

    frame_counter = 0

    while True:
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

        # Save consecutive frames in the output frame folder
        output_frame_path = os.path.join(output_frame_folder, f'output_frame_{frame_counter:04d}.png')
        cv2.imwrite(output_frame_path, mog2_mask_filtered)

        print(f"Saved frame: {output_frame_path}")  # Print the path of each saved frame

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

        # Display the frame
        cv2.imshow('Processed Frame', mog2_mask_filtered)
        cv2.waitKey(30)  # Add a delay to observe the frames

        # Increment frame counter
        frame_counter += 1

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

    print(f"Output frames are saved in: {output_frame_folder}")

    # Release the video capture object and video writer
    cap.release()
    combined_video_writer.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = '/home/raspberry/Desktop/sesan Project-2/capture images/sesan/videos/Envelope_video01.mp4'
output_frame_folder = '/home/raspberry/Desktop/sesan Project-2/capture images/sesan/outputFrames/envelope_frames01'
output_video_path = '/home/raspberry/Desktop/sesan Project-2/capture images/sesan/outputFrames/envelope01.mp4'

# Create the output frame folder if it doesn't exist
os.makedirs(output_frame_folder, exist_ok=True)

process_video(input_video_path, output_frame_folder, output_video_path)




"""
# Example usage
input_video_path = '/home/raspberry/Desktop/sesan Project-2/capture images/bharathi/videos/Envelope_video22.mp4'
output_frame_folder = '/home/raspberry/Desktop/sesan Project-2/capture images/bharathi/outputFrames/envelope_frames'
output_video_path = '/home/raspberry/Desktop/sesan Project-2/capture images/bharathi/outputFrames/envelope.mp4'

process_video(input_video_path, output_frame_folder, output_video_path)

"""


