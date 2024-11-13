import sys
import torch
from ultralytics import YOLO
import ultralytics
import os
import shutil

if __name__ == '__main__':
    print(sys.version)
    print(sys.executable)
    print(torch.__version__)
    print(ultralytics.__version__)

    print(torch.cuda.is_available())

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if device=="cpu":
        print("Warning: Using CPU. This will be slow.")

    # Create a new YOLO model from scratch
    #model = YOLO("yolo11n.yaml")
    # OR
    # Load a pre-existing model
    model = YOLO("models/best.pt")

    print(model)

    # Train the model
    model.train(data='config.yaml', epochs=64)

    # Save the model
    path = model.export()

    objectif = "best.pt"
    new_path = os.path.dirname(path)
    print(new_path)
    source = new_path + "\\" + objectif
    print(source)
    destination = ".\\models"
