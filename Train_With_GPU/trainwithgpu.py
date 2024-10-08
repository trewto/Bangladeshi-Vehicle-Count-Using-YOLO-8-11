import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.get_device_name(0))  # Prints the name of your GPU

#yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0,1
#yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0,1
#>yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0
#yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=4 device=0 amp=False

#main train at 8 oct 2024, take more than 12 hours
#yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=8 device=0

I:\Git\Code-With-Nayeem\Train_With_GPU>yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=8 de
vice=0


#fine tunning at 3.44AM 9 oct 2024
#yolo detect train data=data.yaml model=I:/Git/Code-With-Nayeem/Train_With_GPU/runs/detect/train6/weights/best.pt epochs=10 imgsz=640 batch=8 device=0


yolo detect train data=data.yaml model=I:/Git/Code-With-Nayeem/Train_With_GPU/runs/detect/train6/weights/best.pt epochs=10 imgsz=640 batch=8 device=0