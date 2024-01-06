import cv2
import numpy as np
import os
def track_features(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create random colors for drawing tracks
    color = np.random.randint(0, 255, (100, 3))

    # Read the first frame and detect features
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Counter for frame numbering
    frame_count = 0

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Break the loop if the video is finished
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Combine the frame with the mask
        img = cv2.add(frame, mask)

        # Save the resulting frame to the output folder
        output_path = os.path.join(output_folder, f"tracked_frame_{frame_count:04d}.png")
        cv2.imwrite(output_path, img)

        # Update the previous frame and points for the next iteration
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # Increment the frame counter
        frame_count += 1

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Specify the path to the input video file
    input_video_path = r"F:\sesan's project\videos\VID20231226174804.mp4"

    # Specify the path to the output folder
    output_folder_path = "F:\sesan's project\LucasKanade\EnvelopeLK"

    # Track features and save frames
    track_features(input_video_path, output_folder_path)
