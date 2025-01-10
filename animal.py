from ultralytics import YOLO

# Load a model
model = YOLO("/workspace/ultralytics/yolo8n.pt") # load a pretrained model (recommended for training)

# Train the model

results = model.train(data="/workspace/ultralytics/datasetAinimal/african-wildlife.yaml", epochs=230, imgsz=640)