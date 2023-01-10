import cv2
from imread_from_url import imread_from_url

from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Read image
img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
img = imread_from_url(img_url)

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections
combined_img = yolov8_detector.draw_detections(img)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
cv2.waitKey(0)
