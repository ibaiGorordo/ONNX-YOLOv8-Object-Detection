from ultralytics import YOLO
import json
import argparse

parser = argparse.ArgumentParser(
    prog='CreateONNX',
    description='Create ONNX Model for YOLOv8')
parser.add_argument('-n', '--name', default='yolov8m.pt', help="YOLOv8 pt file")
parser.add_argument('-os', '--opset', default=12, help="ONNX opset version")
args = parser.parse_args()

# Load a model
model_path = "models/{}".format(args.name)
model = YOLO(model_path)

# Test the model
results = model("https://ultralytics.com/images/bus.jpg")
# export the model to ONNX format
model.export(format="onnx", opset=args.opset)
with open('models/{}.json'.format(model_path.split('/')[-1].split('.')[0]), 'w') as fp:
    json.dump(model.names, fp)
