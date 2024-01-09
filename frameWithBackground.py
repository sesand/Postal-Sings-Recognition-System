import cv2
import os

# Function to perform frame differencing and save frames
def process_video(input_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video file
    cap = cv2.VideoCapture(input_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Loop through each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Break the loop if no more frames
        if not ret:
            break

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Save the resulting frame
        output_frame_path = os.path.join(output_folder, f"frame_{frame_count}.png")
        cv2.imwrite(output_frame_path, fg_mask)

        # Display the original and processed frames (optional)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Foreground Mask", fg_mask)

        # Increment frame count
        frame_count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Specify the input video file and output folder
input_video_path = r"C:\Users\admin\Desktop\Postal signs\Videos\VID20231226174804 (1).mp4"
output_frames_folder =r"C:\Users\admin\Desktop\Postal signs\Framediffer&backgroundSubtraction\Envelope"

# Call the function to process the video
process_video(input_video_path, output_frames_folder)
