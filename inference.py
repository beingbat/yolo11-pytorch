import yaml
import torch
import numpy as np
import cv2

from models.model import yolov11
from models.utils import non_max_suppression

INFER_SIZE = (640, 384)

with open(f'coco.yaml','r') as f:
    coco_yml = yaml.safe_load(f)
class_names = list(coco_yml["names"].values())

model = yolov11('small')

wts = torch.load("models/weights/yolo11s.pt")
wts_ = {}
for k in wts:
    wts_[k.replace("model.", "l")] = wts[k]
model.load_state_dict(wts_, strict=True)

model.inference()

img = cv2.imread("zidane.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess(img):
    img = cv2.resize(img, INFER_SIZE)
    img = img.astype(np.double).transpose(2, 0, 1)/255.
    img = torch.from_numpy(img[None, ...])
    return img.float()

# inference
# model.fuse()
preds = model(preprocess(img))[0]
preds = non_max_suppression(
            preds,
            conf_thres=0.5,
            iou_thres=0.7,
            agnostic=False,
            max_det=model.max_det,
            classes=torch.arange(0, model.nc, 1),
        )
# x1, y1, x2, y2 (ltrb), class score, class label
print(preds)

for pred in preds[0].numpy():
    h, w, c = img.shape
    h = h/INFER_SIZE[1]
    w = w/INFER_SIZE[0]
    box = (pred[:4]*(w, h, w, h)).astype(int)
    img = cv2.rectangle(img, 
                        (box[0], box[1]), (box[2], box[3]), 
                        (0, 200, 255), 2)
    img = cv2.putText(img, 
                      f'{class_names[int(pred[-1])]} - {format(pred[4], ".3f")}', 
                      (box[0], box[1]-10),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      1,
                      (0, 200, 255),
                      2, 
                      cv2.LINE_AA)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("zidane_ann.jpg", img)
