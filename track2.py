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


#video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
#print(video_info)

line_color = (0, 255, 0)  # Green color for the line
line_thickness = 2
crossed_count = 0

frame_count = 0
rickshaw_count = 0
tracker = defaultdict(int)  # To track rickshaws passing the line

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(video_info)

def has_crossed_line(start, end, box):
    x1, y1, x2, y2 = box
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    return start[1] <= mid_y <= end[1] and start[0] <= mid_x <= end[0]

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count
    frame_count += 1

    # Perform inference on the frame
    results = model(frame)

    # Loop through the detected objects
    for result in results[0].boxes.data:
        # Extract class ID, confidence, and bounding box coordinates
        class_id, conf, x1, y1, x2, y2 = int(result[5]), result[4], int(result[0]), int(result[1]), int(result[2]), int(result[3])

        # Check if the detected object is a rickshaw (class_id = 0)
        if class_id == 0:
            box_id = (x1, y1, x2, y2)
            # Check if the rickshaw has crossed the line
            if has_crossed_line(START, END, box_id):
                # Increment the rickshaw count if not already counted
                if tracker[frame_count] == 0:
                    rickshaw_count += 1
                    tracker[frame_count] = 1

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the counting line
    cv2.line(frame, START, END, (0, 0, 255), 2)

    # Display the rickshaw count on the frame
    cv2.putText(frame, f"Rickshaw Count: {rickshaw_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the processed frame
    #cv2.imshow('Rickshaw Detection', frame)
    cv2.imwrite('frame.jpg', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()

print(f"Total Rickshaws Counted: {rickshaw_count}")