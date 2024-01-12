import cv2
import os
from datetime import datetime, timedelta

# Set up video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

# Set up background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Create a folder to store the subtracted frames
output_folder = '/home/raspberry/Desktop/Sesan-Project/sample programs/frameAndBackground/envelope frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the desired time duration (in seconds)
desired_duration = 4

# Initialize variables
start_time = datetime.now()
frames = []

while (datetime.now() - start_time).total_seconds() < desired_duration:
    # Capture a frame
    ret, frame = video_capture.read()

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Show the frame with the background subtracted
    cv2.imshow('Background Subtraction', fgmask)

    # Append the subtracted frame to the list
    frames.append(fgmask.copy())

    # Save the subtracted frame
    output_file = os.path.join(output_folder, f'subtracted_frame_{len(frames)}.png')
    cv2.imwrite(output_file, fgmask)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('envelope.mp4', fourcc, 20.0, (640, 480))  # Adjust the resolution as needed

# Write the frames to the video
for frame in frames:
    # Convert grayscale to BGR before writing to the video
    output_video.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

# Release the video writer
output_video.release()

print(f'Capture completed. Subtracted frames saved in: {output_folder}')
print('Video saved as: output_video.mp4')





