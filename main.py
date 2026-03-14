import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sklearn.cluster import KMeans

# --- SETTINGS ---
SOURCE_VIDEO_PATH = "game.mp4"
TARGET_VIDEO_PATH = ("Cristiano.mp4")
MODEL_PATH = "yolov8n.pt"

# Add your player ID-name mappings here.
# The key is the unique tracker ID, and the value is the player's name.
PLAYER_NAMES = {
    185: "Cristiano Ronaldo"
    # You can add more players here like this:
    # 210: "Another Player Name",
}
# Define Team names and the target color for one team (e.g., Portugal's red in BGR)
TEAM_NAMES = ["Portugal", "Spain"]
PORTUGAL_TARGET_COLOR_BGR = (0, 0, 255)

# --- SCRIPT STARTED ---
print("--- SCRIPT STARTED ---")

# Load the YOLO model
model = YOLO(MODEL_PATH)
print("Checkpoint 1: YOLO model loaded successfully.")

# Instantiate the ByteTrack tracker
tracker = sv.ByteTrack()


# Helper Function
def get_dominant_color(image, k=1):
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = np.uint8(center[0])
    return tuple(map(int, dominant_color))


# --- Video Processing ---
try:
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Checkpoint 2: Video info loaded. Video resolution: {video_info.width}x{video_info.height}")
except Exception as e:
    print(f"ERROR: Could not load video info. Check if '{SOURCE_VIDEO_PATH}' exists and is a valid video file.")
    print(f"Details: {e}")
    exit()

frame_number = 0
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    print("Checkpoint 3: Output video file opened. Starting frame processing loop...")

    for frame in sv.get_video_frames_generator(SOURCE_VIDEO_PATH):
        frame_number += 1
        print(f"--> Processing frame {frame_number}...")

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]

        tracked_objects = tracker.update_with_detections(detections)
        print(f"    - Detected and tracked {len(tracked_objects.xyxy)} players.")

        if len(tracked_objects.xyxy) > 1:
            print("    - Condition met: More than one player found. Attempting team assignment.")
            player_colors = []
            for bbox in tracked_objects.xyxy:
                x1, y1, x2, y2 = map(int, bbox)
                player_image = frame[y1:y2, x1:x2]
                if player_image.size > 0:
                    dominant_color = get_dominant_color(player_image)
                    player_colors.append(dominant_color)

            if len(player_colors) > 1:
                kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(player_colors)
                team_ids = kmeans.labels_

                cluster_centers = kmeans.cluster_centers_
                portugal_cluster_id = np.argmin(
                    np.linalg.norm(cluster_centers - PORTUGAL_TARGET_COLOR_BGR, axis=1)
                )
                team_mapping = {
                    portugal_cluster_id: "Portugal",
                    1 - portugal_cluster_id: "Spain"
                }

                annotated_frame = frame.copy()
                for i, bbox in enumerate(tracked_objects.xyxy):
                    x1, y1, x2, y2 = map(int, bbox)
                    team_id = team_ids[i]
                    tracker_id = tracked_objects.tracker_id[i]

                    team_name = team_mapping[team_id]
                    team_color = (0, 0, 255) if team_name == "Portugal" else (255, 255, 255)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), team_color, 2)

                    # 💡 MODIFIED LINE: This gets the name from the dictionary or defaults to the ID
                    player_name = PLAYER_NAMES.get(tracker_id, f"Player #{tracker_id}")
                    label = f"{player_name} ({team_name})"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

                sink.write_frame(annotated_frame)
                print("    - Frame was ANNOTATED and written to the output file.")
            else:
                sink.write_frame(frame)
                print("    - Frame was NOT annotated (not enough players with color) and written to the output file.")
        else:
            sink.write_frame(frame)
            print("    - Frame was NOT annotated (not enough players tracked) and written to the output file.")

print(f"\n--- SCRIPT FINISHED after processing {frame_number} frames. ---")