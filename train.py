from ultralytics import YOLO

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 80  # number of layers to freeze
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")


model = YOLO("yolo11n.pt") 
model.add_callback("on_train_start", freeze_layer)

#yolo detect train data=data.yaml model= yolo11n.pt epochs=25 imgsz=640 batch=12 device=0 –freeze=80
model.train(data="I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/data.yaml", epochs =10, imgsz=640, batch=12, device=0)