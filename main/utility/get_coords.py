import numpy as np
import cv2
from math import floor
import torch
import open3d


def get_strides(img_shape, ideal_img_shape):
    strides_ratio = (img_shape[0]/ideal_img_shape[0], img_shape[1]/ideal_img_shape[1])
    if any (strides < 0.5 for strides in strides_ratio):
        raise Exception("Image/PointCloud is too small for YOlov5 topView to detect")
    else:
        strides_ratio = (int(round(strides_ratio[0])), int(round(strides_ratio[1])))
    return strides_ratio

def scale_coord(coords, scale:tuple, offset:tuple):
    coords[:,0] = coords[:,0]*scale[0] + offset[0]
    coords[:,1] = coords[:,1]*scale[1] + offset[1]
    return coords

def scale_pred_to_xy_point_cloud(pred:np.ndarray, stepsize, min_x:float=0.0, max_y:float=0.0):
    """
    Params
        pred    : [np.ndarray] of shape predictions from yolo
        stepsize: [float] stepsize used to generate this image
        min_x   : [float] minimum x value of point cloud
        min_y   : [float] minimum y value of point cloud
    
    Returns
        x,y,pred     : [np.ndarray] center point of each prediction + prediction
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rtn_arr = torch.zeros((pred.shape[0],3), device=device)
    xy_gpu = torch.from_numpy(pred).to(device)
    
    # x_center and y_center of Bounding box 
    rtn_arr[:,0] = (torch.add(xy_gpu[:,0],xy_gpu[:,2])/2*stepsize)+min_x
    rtn_arr[:,1] = (torch.add(xy_gpu[:,1],xy_gpu[:,3])/2*stepsize)+max_y
    rtn_arr[:,2] = xy_gpu[:,4]
    #print(xy_gpu[:,4])
    rtn_arr = rtn_arr.cpu().numpy()
    return rtn_arr


def draw_coord_on_img(img, coords, circle_size=2):
    img = img.copy()
    coords = coords.astype(int)
    for coord in coords:
        cv2.circle(img,(coord[0],coord[1]), circle_size, (242,2,2),3)
    return img


def draw_coord_on_img_with_pred(img, uv_coords_pred, height, circle_size = 2): #(xc,yc,confidence,label)
    uv_coords_pred = uv_coords_pred.astype(int)
    if not uv_coords_pred.size:
        return img
    red = (255,0,0)
    blue = (0,0,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for coord in uv_coords_pred:
        if coord[3]:
            color = red
        else:
            color = blue
        img = cv2.circle(img,(coord[0],coord[1]), circle_size, color,5)
        img = cv2.putText(img, str(coord[2]) ,(coord[0]+20,coord[1]+2), font, 1,color,2,cv2.LINE_AA) # Draw Confidence
        img = cv2.putText(img, f"{(coord[0],coord[1])}" ,(coord[0]-50,coord[1]-50), font, 1,color,4,cv2.LINE_AA) # Draw (x_or_y, z)
    img = cv2.putText(img, f"H={height}", (50,50),font, 1,red,2,cv2.LINE_AA)
    return img

