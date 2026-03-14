import os
from ultralytics import YOLO
import cv2

# Load a pre-trained YOLO model (e.g., yolov8n.pt is a small, fast one)
model = YOLO("yolov8n.pt")

# --- SETTINGS ---
INPUT_FOLDER = "frames"
OUTPUT_FOLDER = "frames_detected"
# ----------------

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Created output folder: {OUTPUT_FOLDER}")

# Get all the image file names from the input folder
image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.jpg')])

print(f"Found {len(image_files)} frames to process...")

# Loop through each image file
for image_name in image_files:
    # Build the full path to the input image
    image_path = os.path.join(INPUT_FOLDER, image_name)

    # Run detection on the image
    # The model.predict() function returns a list of results
    results = model.predict(source=image_path, save=False)  # We set save=False to draw our own boxes

    # The result object contains the plotted image with boxes drawn on it
    # We access the first result since we process one image at a time
    annotated_frame = results[0].plot()

    # Build the full path for the output image
    output_path = os.path.join(OUTPUT_FOLDER, image_name)

    # Save the new image with the detections
    cv2.imwrite(output_path, annotated_frame)

    print(f"  - Processed and saved {image_name}")

print("\n--- Object detection complete! ---")
print(f"Annotated frames are saved in the '{OUTPUT_FOLDER}' folder.")