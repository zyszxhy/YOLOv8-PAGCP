import torch
from ultralytics import YOLO
#from ultralytics.nn.modules import C2f_v2
from ultralytics.yolo.utils.torch_utils import model_info
def main():

    model = YOLO("runs/detect/train4/weights/best.pt")
    # model = YOLO("shufflenetv2-v8.yaml")
    # model.train(data='NWPU.yaml', epochs=120, imgsz=640, batch=16, lr0=0.01,pruning=False)
    results = model.val(data='NWPU.yaml' ,fuse=True)  # predict on an image

if __name__ == '__main__':
    main()


