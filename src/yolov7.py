import cv2
import torch
import numpy as np
import os

#os.chdir("src")
#print(os.getcwd())

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device, TracedModel

#weight_loc = './weights/final.pt' # Modelin ağırlık dosyasının konumu
#weight_loc = './weights/lasss.pt' # Modelin ağırlık dosyasının konumu
weight_loc = './weights/best.pt' # Modelin ağırlık dosyasının konumu
image_size = 896 # Modelin işleyeceği görüntü boyutu
_device = '' # Kullanılacak cihaz (GPU için boş bırakın)
trace = False
# Initialize
device = select_device(_device) # GPU veya CPU seçimi
half = False #device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weight_loc, map_location=device)  # load FP32 model
stride = int(model.stride.max())
imgsz = check_img_size(image_size, s=stride) # Seçilen görüntü boyutu 32'nin katı mı kontrol et

if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()  # to FP16

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

def analyze(img0, save_loc=None, _augment=False, conf_thres=0.4, iou_thres=0.5, agnostic_nms=False):
    with torch.no_grad():
        # TODO test if transpose blow is better
        # Read Image - LoadImages
        img = letterbox(img0, imgsz, stride=stride)[0]
        predictions = []
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=_augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
        #t2 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    x0 = int(xyxy[0])
                    y0 = int(xyxy[1])
                    x1 = int(xyxy[2])
                    y1 = int(xyxy[3])
                    if int(cls) == 0:
                        cls = 1
                    elif int(cls) == 1:
                        cls = 0
                    else:
                        print("Error")
                        
                    predictions.append([int(cls), "-1", x0, y0, x1, y1])

        return predictions

