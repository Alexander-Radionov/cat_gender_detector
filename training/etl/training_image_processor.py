import os
import cv2
import supervision as sv
from pathlib import Path
from ultralytics import YOLO
from definitions import ROOT_DIR,  EXCEPTIONS_FILE_NAME


class TrainingImageProcessor:
    def __init__(self, model_path, conf=0.5, verbose=False):
        self.exceptions_file_name = EXCEPTIONS_FILE_NAME
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.confidence = conf
        self.verbose = verbose
        self.category_dict = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }

    def detect_objects(self, image):
        results = self.model(source=image, conf=self.confidence, verbose=self.verbose)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    @staticmethod
    def detections_to_yolo_format(detections, img_h, img_w) -> list[tuple]:
        yolo_coords = []
        for coords in detections.xyxy:
            ymin, xmin, ymax, xmax = coords.tolist()
            box_height = (ymax - ymin) / img_h
            box_width = (xmax - xmin) / img_w
            center_x = (xmin + (xmax - xmin) / 2) / img_w  # Fixed calculation
            center_y = (ymin + (ymax - ymin) / 2) / img_h  # Fixed calculation
            yolo_coords.append((center_x, center_y, box_width, box_height))
        return yolo_coords

    @staticmethod
    def write_detections_to_labels_file(yolo_detections, gt_labels, image_path):
        labels_folder_path = Path(ROOT_DIR).parent / 'data' / 'labels'
        os.makedirs(str(labels_folder_path), exist_ok=True)
        label_file_path = str(labels_folder_path / Path(image_path).name.replace('.png', '.txt'))
        with open(label_file_path, 'w') as f:
            for image_coords, image_label in zip(yolo_detections, gt_labels):
                f.write(f"{image_label} {' '.join([str(c) for c in image_coords])}\n")

    def process_training_image(self, image_path, label) -> bool:
        try:
            image = cv2.imread(image_path)
            if image is None:
                with open(self.exceptions_file_name, 'a') as f:
                    f.write(f"Image path: {image_path}. Error: Could not read image file\n")
                return False
                
            detections = self.detect_objects(image)
            # Filter detections to cat class if available; fall back to any detection
            try:
                cat_detections = detections[detections.class_id == 15]
            except Exception:
                cat_detections = detections
            
            if len(cat_detections) == 0:
                with open(self.exceptions_file_name, 'a') as f:
                    f.write(f"Image path: {image_path}. Error: No cats detected\n")
                return False
                
            img_height, img_width = image.shape[:2]  # Fixed order
            yolo_detections = self.detections_to_yolo_format(cat_detections, img_height, img_width)
            
            if len(yolo_detections) > 1:
                with open(self.exceptions_file_name, 'a') as f:
                    f.write(f"Image path: {image_path}. Error: More than 1 cat detected. Skipping\n")
                return False
            
            self.write_detections_to_labels_file(yolo_detections, [label], image_path)
            return True
            
        except Exception as e:
            with open(self.exceptions_file_name, 'a') as f:
                f.write(f"Image path: {image_path}. Error: {str(e)}\n")
            return False
