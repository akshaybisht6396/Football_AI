import cv2
import os

# --- SETTINGS ---
INPUT_FOLDER = 'frames_detected'  # The folder with your processed images
OUTPUT_VIDEO_PATH = 'output_video.mp4'
FPS = 30  # Frames Per Second for the output video
# ----------------

# Get all the image file names and sort them numerically
try:
    image_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.jpg')],
        key=lambda x: int(os.path.splitext(x.split('_')[1])[0])
    )
    if not image_files:
        print(f"Error: No .jpg files found in the '{INPUT_FOLDER}' directory.")
        exit()
except (IndexError, ValueError):
    print(f"Error: Files in '{INPUT_FOLDER}' are not in the expected 'frame_NUM.jpg' format.")
    exit()

# Read the first image to get the frame dimensions (width, height)
first_image_path = os.path.join(INPUT_FOLDER, image_files[0])
frame = cv2.imread(first_image_path)
if frame is None:
    print(f"Error: Could not read the first image at {first_image_path}")
    exit()
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'XVID'
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (width, height))

print(f"Creating video from {len(image_files)} frames...")

# Loop through each image and write it to the video
for image_name in image_files:
    image_path = os.path.join(INPUT_FOLDER, image_name)
    frame = cv2.imread(image_path)
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print("\n--- Video creation complete! ---")
print(f"Video saved as '{OUTPUT_VIDEO_PATH}'")