import cv2
import os

def frame_difference(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the first frame
    ret, prev_frame = cap.read()

    # Counter for frame numbering
    frame_count = 0

    while True:
        # Read the current frame
        ret, curr_frame = cap.read()

        # Break the loop if the video is finished
        if not ret:
            break

        # Calculate the absolute difference between frames
        diff_frame = cv2.absdiff(prev_frame, curr_frame)

        # Convert the difference frame to grayscale
        diff_frame_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

        # Threshold the difference frame
        _, thresh = cv2.threshold(diff_frame_gray, 30, 255, cv2.THRESH_BINARY)

        # Save the resulting frame to the output folder
        output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(output_path, thresh)

        # Update the previous frame for the next iteration
        prev_frame = curr_frame

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Specify the path to the input video file
    input_video_path = r"F:\sesan's project\videos\VID20231226174926.mp4"

    # Specify the path to the output folder
    output_folder_path = r"F:\sesan's project\frame differencing\LetterFrame"

    # Perform frame differencing and save frames
    frame_difference(input_video_path, output_folder_path)
