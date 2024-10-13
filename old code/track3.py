import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
from supervision.draw.color import ColorPalette

SOURCE_VIDEO_PATH = './Processing/Katabon_Intersection_720p.mp4'


# Open the video file
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

#width=1280, height=720, fps=25, total_frames=45612
START = (0, 460)
END = (1178, 460)

# Load the YOLOv8 model
model = YOLO('best.pt')
# Store the track history
track_history = defaultdict(lambda: [])

# Create a dictionary to keep track of objects that have crossed the line
crossed_objects = {}


#video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
#print(video_info)
#VideoInfo(width=1280, height=720, fps=25, total_frames=45612)

line_color = (0, 255, 0)  # Green color for the line
line_thickness = 2
crossed_count = 0

frame_count = 0
rickshaw_count = 0
tracker = defaultdict(int)  # To track rickshaws passing the line

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(video_info)


with sv.VideoSink("output_single_line.mp4", video_info) as sink:
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        #processing the frame
        frame_count += 1
        if frame_count< 1000+ int(cap.get(cv2.CAP_PROP_FPS) * 270):
            #print(frame_count)
            continue
        if frame_count > 7749+500:#+500 
            break
        if not success:
            print("Error: Could not read frame.")
            break
        print(f"Processing frame {frame_count}")

        results = model.track(frame, classes=[0], persist=True, save=False, tracker="bytetrack.yaml")

        if results is None or len(results) == 0:
            print("No detection results for this frame.")
            #continue

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id
        annotated_frame = results[0].plot()

        if track_ids is None:
            print("No track IDs found for this frame.")
            continue
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()

            # Visualize the results on the frame
            #annotated_frame = results[0].plot()

            # Plot the tracks and count objects crossing the line
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 6000:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Check if the object crosses the line

                
                if START[0] < x < END[0] and abs(y - START[1]) < 30:  # Assuming objects cross horizontally
                    if track_id not in crossed_objects:
                        crossed_objects[track_id] = True

                    # Annotate the object as it crosses the line
                    cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

            # Draw the line on the frame
            cv2.line(annotated_frame, START, END, (0, 255, 0), 2)

        # Write the count of objects on each frame
        count_text = f"Objects crossed: {len(crossed_objects)}"
        cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame with annotations to the output video
        sink.write_frame(annotated_frame)
        
        
# Release the video capture
cap.release()
print("Processing complete.")

        