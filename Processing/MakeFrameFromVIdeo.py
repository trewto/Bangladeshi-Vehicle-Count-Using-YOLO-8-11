import cv2
import os

# Path to the video file on your desktop
video_path = 'C:/Users/User/Downloads/YVIDEO/pervez/Abul_Hotel_Cam1_1_10fps.mp4'  # Ensure this is correct

# Path to the folder where you want to save the frames
output_folder = './parvez/abulh/'  # Ensure this is correct

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the frame rate (frames per second)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Error: FPS could not be determined.")
    exit()

print(f"FPS: {fps}")
#exit()
# Define the interval (in seconds) to save frames
interval = 7
frame_interval = int(fps * interval)

frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends or cannot read frame

    # Save frame at the specified interval
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")
        saved_frame_count += 1

    frame_count += 1

# Release the video capture object
cap.release()
print(f"{saved_frame_count} frames extracted")
print("Current working directory:", os.getcwd())
