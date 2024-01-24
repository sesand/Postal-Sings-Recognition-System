import cv2
import os
from datetime import datetime, timedelta

# Set up video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Create a folder to store the subtracted frames
output_folder = 'subtracted_frames002'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the capture duration (in seconds)
capture_duration = 6
end_time = datetime.now() + timedelta(seconds=capture_duration)

frame_count = 0

# Specify the output video file
output_video_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on your system
output_video = cv2.VideoWriter(output_video_file, fourcc, 20.0, (640, 480))  # Adjust frame size and FPS as needed

while True:
    # Capture a frame
    ret, frame = video_capture.read()

    # Check if the frame is successfully captured
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Display the frame with the background subtracted
    cv2.imshow('Video Capture', frame)
    cv2.imshow('Background Subtraction', fgmask)

    # Save the subtracted frame as an image (optional)
    output_file = os.path.join(output_folder, f'subtracted_frame_{frame_count}.png')
    cv2.imwrite(output_file, fgmask)

    # Write frame to the video file
    output_video.write(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))  # Convert to BGR before writing

    frame_count += 1

    # Check if the specified duration has elapsed
    if datetime.now() >= end_time:
        break

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, close windows, and release the video writer
video_capture.release()
output_video.release()
cv2.destroyAllWindows()

print(f'Capture completed. Subtracted frames saved in: {output_folder}')
print(f'Output video file: {output_video_file}')
