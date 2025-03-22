import cv2
from typing import List, Tuple

def yolo_detect(image) -> List[Tuple[int,int,int,int,str]]:
    """
    Dummy YOLO detection function. In practice, run your actual YOLO model.
    Returns a list of bounding boxes (x1, y1, x2, y2) and a label for the column.
    """
    h, w = image.shape[:2]
    # Example: 4 bounding boxes for Test Name, Value, Unit, Reference Value
    # (in the center, just for demonstration)
    boxes = [
        (int(0.1*w), int(0.1*h), int(0.4*w), int(0.2*h), "Test Name"),
        (int(0.5*w), int(0.1*h), int(0.7*w), int(0.2*h), "Value"),
        (int(0.1*w), int(0.3*h), int(0.4*w), int(0.4*h), "Unit"),
        (int(0.5*w), int(0.3*h), int(0.7*w), int(0.4*h), "Ref Value"),
    ]
    return boxes
