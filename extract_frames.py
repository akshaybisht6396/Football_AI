import cv2
import os

# --- SETTINGS ---
VIDEO_PATH = 'game.mp4'
OUTPUT_FOLDER = 'frames'
# ----------------

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        break  # End of video

    # Save the frame as a JPG file
    frame_filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

    # Print progress
    if frame_count % 100 == 0:
        print(f"Extracted {frame_count} frames...")

print(f"\nExtraction complete. Total frames saved: {frame_count}")
cap.release()