from ultralytics import YOLO

model = YOLO("best.pt")  # Load an official Detect model

results = model.track("https://www.youtube.com/watch?v=n6ShWPom7ys", show=True, tracker="bytetrack.yaml")  # with ByteTrack