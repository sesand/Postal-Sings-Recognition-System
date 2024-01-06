import cv2
import os

def detect_foreground(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Counter for frame numbering
    frame_count = 0

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Break the loop if the video is finished
        if not ret:
            break

        # Apply the background subtractor to get the foreground mask
        fg_mask = bg_subtractor.apply(frame)

        # Save the resulting foreground mask to the output folder
        output_path = os.path.join(output_folder, f"foreground_{frame_count:04d}.png")
        cv2.imwrite(output_path, fg_mask)

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Specify the path to the input video file
    input_video_path = r"F:\sesan's project\videos\VID20231226174926.mp4"

    # Specify the path to the output folder
    output_folder_path = r"F:\sesan's project\BackgroundSubtraction\LetterBS"

    # Detect foreground objects and save masks
    detect_foreground(input_video_path, output_folder_path)
