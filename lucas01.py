
# 001 lucas kanade video tracking

"""
import cv2
import numpy as np

def track_and_save(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=10)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect features in the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow only if there are valid feature points
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Check if points were successfully tracked
            if p1 is not None:
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                img = cv2.add(frame, mask)

                # Write the frame to the output video file
                out.write(img)

                cv2.imshow('Frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "/home/raspberry/Desktop/Sesan-Project/background subtraction/output_video.mp4"
    output_video_path = "/home/raspberry/Desktop/Sesan-Project/Lucas Kanade Method/lucasKanade03.mp4"
    track_and_save(input_video_path, output_video_path)

"""

# 002 lucas kanade video tracking
"""
import cv2
import numpy as np

def track_and_save(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Initial feature detection in the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow only if there are valid feature points
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Check if points were successfully tracked
            if p1 is not None:
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                img = cv2.add(frame, mask)

                # Write the frame to the output video file
                out.write(img)

                cv2.imshow('Frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Update previous frame and feature points only when necessary
        old_gray = frame_gray.copy()
        if len(good_new) > 0:
            p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "/home/raspberry/Desktop/Sesan-Project/background subtraction/output_video.mp4"
    output_video_path = "/home/raspberry/Desktop/Sesan-Project/Lucas Kanade Method/tracked_video01.mp4"
    track_and_save(input_video_path, output_video_path)

"""

# 003 lucas kanade video tracking

"""
import cv2
import numpy as np

def track_and_save(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=50, qualityLevel=0.01, minDistance=5, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect features in the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow only if there are valid feature points
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Check if points were successfully tracked
            if p1 is not None:
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                img = cv2.add(frame, mask)

                # Write the frame to the output video file
                out.write(img)

                cv2.imshow('Frame', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "/home/raspberry/Desktop/Sesan-Project/background subtraction/output_video.mp4"
    output_video_path = "/home/raspberry/Desktop/Sesan-Project/Lucas Kanade Method/lucasKanade02.mp4"
    track_and_save(input_video_path, output_video_path)
    
"""

# 004 lucas kanade video tracking
# apply a median filter 

import cv2
import numpy as np

def track_and_save_hands(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Parameters for hand feature detection
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
    """
	# Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    """
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply median blur to reduce noise
        frame = cv2.medianBlur(frame, 7)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features in the frame (hands)
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply median blur to reduce noise
            frame = cv2.medianBlur(frame, 5)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow only if there are valid feature points (hands)
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, frame_gray, p0, None, **lk_params)

                # Check if points were successfully tracked
                if p1 is not None:
                    # Select good points
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # Draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                    img = cv2.add(frame, mask)

                    # Write the frame to the output video file
                    out.write(img)

                    cv2.imshow('Frame', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            # Update feature points for the next frame
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = "/home/raspberry/Desktop/Sesan-Project/background subtraction/output_video.mp4"
    output_video_path = "/home/raspberry/Desktop/Sesan-Project/Lucas Kanade Method/lucas_kanade007.mp4"
    track_and_save_hands(input_video_path, output_video_path)

"""
if __name__ == "__main__":
    input_folder_path = "/home/raspberry/Desktop/Sesan-Project/background subtraction/subtracted_frames002"
    output_folder_path = "/home/raspberry/Desktop/Sesan-Project/Lucas Kanade Method/lucas kanade frames002"
    track_frames(input_folder_path, output_folder_path)
"""

# lucas kanade video tracking

"""
if __name__ == "__main__":
    input_video_path = "/home/raspberry/Desktop/Sesan-Project/background subtraction/output_video.mp4"
    output_video_path = "/home/raspberry/Desktop/Sesan-Project/Lucas Kanade Method/tracked_video.mp4"
    track_and_save(input_video_path, output_video_path)
"""
