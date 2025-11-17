import os
import cv2
import torch
import numpy as np
import contextlib
import threading
import open3d as o3d
import logging
logger = logging.getLogger("my-app")
logger.setLevel(logging.INFO)


class TryExcept(contextlib.ContextDecorator):
    # YOLOv5 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='default message here'):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            print(f'{self.msg}: {value}')
        return True

def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper

def path_exists_assert(save_dir:str):
    assert os.path.exists(save_dir), f"the path or file [{save_dir}] does not exist"

    
class TPH_IO:
    def __init__(self):
        pass
    
    def save_pcd_compressed(self, pcd:o3d.t.geometry.PointCloud, save_dir:str, filename:str):
        file_dir = os.path.join(save_dir, f"{filename}.ply")
        path_exists_assert(save_dir)
        # pcd.tofile(file_dir)
        o3d.io.write_point_cloud(file_dir, pcd, write_ascii=False, print_progress=True)
        return

    def load_pcd_compressed(self, load_dir:str, filename:str)->o3d.t.geometry.PointCloud:
        path_exists_assert(load_dir)
        # np_arr = np.fromfile(load_dir, dtype=np.float32)
        # device = o3d.core.Device("CPU:0")
        # dtype = o3d.core.float32
        # pcd = o3d.t.geometry.PointCloud(np_arr)
        # pcd.cuda()
        
        file_dir = os.path.join(load_dir, f"{filename}.ply")
        pcd = o3d.io.read_point_cloud(file_dir)
        print(len(pcd.points))
        return pcd
    
    
    @threaded
    def save_img(self, img:np.ndarray, save_dir:str, filename:str, bool_gray:bool=False):
        path_exists_assert(save_dir)
        if bool_gray:
            cv2.imwrite(f"{save_dir}/{filename}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        else:
            cv2.imwrite(f"{save_dir}/{filename}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @threaded
    def load_img(self, load_dir:str, bool_gray:bool)->np.ndarray:
        path_exists_assert(load_dir)
        if bool_gray:
            array = cv2.imread(load_dir, cv2.IMREAD_GRAYSCALE)
        else:
            array = cv2.imread(load_dir, cv2.IMREAD_COLOR)
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return array
    
    
    @threaded
    def save_npy_img(self, img_arr:np.ndarray, save_dir:str, filename:str):
        path_exists_assert(save_dir)
        save_img_dir = os.path.join(save_dir, f"{filename}.bin")
        img_arr.tofile(save_img_dir)
        
    # @threaded
    def load_npy_img(self, load_dir:str, np_dtype:np.dtype)->np.ndarray:
        path_exists_assert(load_dir)
        array = np.fromfile(load_dir, dtype=np_dtype)
        return array