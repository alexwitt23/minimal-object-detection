import cv2
import torch
import numpy as np

from src.model import detector


def normalize(
    img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
) -> np.ndarray:

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator

    return img


save_path = "/tmp/img.jpg"
image = cv2.imread(
    "/home/alex/Desktop/projects/minimal-object-detector/src/train/data/images/2020-Toyota-86-GT-TRD-Wheels.jpg"
)
image_ori = cv2.resize(image, (512, 512))
image = normalize(image_ori)


model = detector.Detector(timestamp="2021-04-22T11.25.25")
model.eval()

with torch.no_grad():
    image = torch.Tensor(image)
    if torch.cuda.is_available():
        image = image.cuda()
    boxes = model.get_boxes(image.permute(2, 0, 1).unsqueeze(0))

for box in boxes[0]:
    print(box)
    box = (box.box * torch.Tensor([512] * 4)).int().tolist()
    cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

cv2.imwrite(save_path, image_ori)
