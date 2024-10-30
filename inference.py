import torch
from models.model import yolov11

yolo_network = yolov11('small')

wts = torch.load("models/weights/yolo11s.pt")
wts_ = {}
for k in wts:
    wts_[k.replace("model.", "l")] = wts[k]

yolo_network.load_state_dict(wts_, strict=True)
print(yolo_network)
# add code here
