import cv2
from moviepy import VideoFileClip, concatenate_videoclips
from ultralytics import YOLO

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "highlights.mp4"

model = YOLO("yolo11n.pt")
IMPORTANT_OBJECTS = {"person", "sports ball"}

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = 0
highlight_times = []
frame_skip = max(1, int(fps))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_number % frame_skip == 0:
        time_sec = frame_number / fps
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]
                confidence = float(box.conf[0])

                if object_name in IMPORTANT_OBJECTS and confidence > 0.5:
                    print(f"Detected {object_name} at {time_sec:.2f}s")
                    highlight_times.append(time_sec)
                    break

    frame_number += 1

cap.release()

filtered_times = []
for timestamp in highlight_times:
    if not filtered_times or timestamp - filtered_times[-1] > 5:
        filtered_times.append(timestamp)

video = VideoFileClip(VIDEO_PATH)
clips = []

for timestamp in filtered_times[:10]:
    start = max(0, timestamp - 3)
    end = min(video.duration, timestamp + 3)
    clips.append(video.subclipped(start, end))

if clips:
    final = concatenate_videoclips(clips)
    final.write_videofile(OUTPUT_PATH)
    print(f"Done. Saved highlight video as {OUTPUT_PATH}")
else:
    print("No highlights found.")

