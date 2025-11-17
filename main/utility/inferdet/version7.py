import sys
sys.path.insert(0, '/root/sdp_tph/main/yolov7/segment')
from inferenceFrombatch import *


class Detect7Bro:
    def __init__(self, source_img_dir, yaml_conf_dir, weights_pth, batch_size, image_size, save_mask_dir, save_overlays_dir=None, conf_thres=0.25, iou_thres=0.6, max_det=300):
        self.run = Detectv7(source_img_dir, yaml_conf_dir, weights_pth, batch_size, image_size, save_mask_dir, save_overlays_dir=None, conf_thres=0.25, iou_thres=0.6, max_det=300)
    
    