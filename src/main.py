
import cv2
import torch

import detector


image = cv2.imread("/home/alex/datasets/coco/images/val2017/000000000285.jpg")
image = cv2.resize(image, (512, 512))


model = detector.Detector(num_classes=10, confidence=.01)


with torch.no_grad():
    boxes = model(torch.Tensor(image).permute(2, 0, 1).unsqueeze(0))

print(boxes)